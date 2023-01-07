from typing import List, Optional
import sys
sys.path.append('./python')
from needle.autograd import Tensor
from needle import ops, array_api
import needle.init as init
from needle.backend_ndarray.ndarray import BackendDevice
import needle.nn as nn

import numpy as np
from tqdm.auto import tqdm
import math


class UnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, device=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch, device=device)
        if up:
            self.conv1 = nn.Conv(2 * in_ch, out_ch, 3, padding=1, device=device)
            self.transform = nn.ConvTranspose(out_ch, out_ch, 4, 2, 1, device=device)
        else:
            self.conv1 = nn.Conv(in_ch, out_ch, 3, padding=1, device=device)
            self.transform = nn.MaxPool(2)
        self.conv2 = nn.Conv(out_ch, out_ch, 3, padding=1, device=device)
        self.bnorm1 = nn.BatchNorm2d(out_ch, device=device)
        self.bnorm2 = nn.BatchNorm2d(out_ch, device=device)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.reshape(time_emb.shape + (1, 1)).broadcast_to(h.shape)

        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class Unet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self, device: Optional[BackendDevice] = None):
        super().__init__()
        image_channels = 3
        down_channels = (32, 64, 128)
        up_channels = down_channels[::-1]
        out_dim = 1
        time_emb_dim = 16

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim, device=device),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv(image_channels, down_channels[0], 3, device=device)

        # Downsample
        self.downs = []
        # Upsample
        self.ups = []

        for i in range(len(down_channels) - 1):
            self.downs.append(
                UnetBlock(down_channels[i], down_channels[i + 1], time_emb_dim, device=device)
            )

        for i in range(len(up_channels) - 1):
            self.ups.append(
                UnetBlock(up_channels[i], up_channels[i + 1], time_emb_dim, up=True, device=device)
            )

        self.output = nn.Conv(up_channels[-1], 3, out_dim, device=device)

    def forward(self, x: Tensor, timestep: Tensor) -> Tensor:
        # x.shape = (B, C, H, W)
        # timestep.shape = (B,)
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []

        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = ops.stack((x, residual_x), axis=2).reshape(
                (x.shape[0], 2*x.shape[1], *x.shape[2:])
            )
            x = up(x, t)

        return self.output(x)


class SinusoidalPosEmb(nn.Module):
    DENOM_BASEMENT = 10000

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        # x.shape = (batch_size,)
        # Returns emb with shape = (batch_size, self.dim)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.DENOM_BASEMENT) / (half_dim - 1)
        emb = ops.exp(Tensor(range(half_dim), device=device) * -emb)
        emb = x.broadcast_to((x.size, 1)) @ emb.broadcast_to((1, emb.size))
        emb = ops.stack(
            (ops.sin(emb), ops.cos(emb)), axis=2
        ).reshape((x.size, self.dim))
        return emb


class Diffusion:
    def __init__(
        self,
        model,
        timesteps,
        beta_schedule="linear",
        loss_type="l1",
        device=None
    ):
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps, device=device)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps, device=device)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.denoise_model = model
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "l2":
            self.loss_fn = nn.L2Loss()
        else:
            raise NotImplementedError(f"Unknown loss {loss_type}")

        self.timesteps = timesteps

        alphas = 1.0 - self.betas
        alphas_cumprod = array_api.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = Tensor(
            np.pad(alphas_cumprod.numpy()[:-1], (1, 0), constant_values=1.0),
            device=device,
            requires_grad=False
        )
        self.sqrt_recip_alphas = Tensor(
            (1.0 / alphas) ** (1 / 2),
            device=device,
            requires_grad=False
        )
        self.sqrt_alphas_cumprod = Tensor(
            (alphas_cumprod) ** (1 / 2),
            device=device,
            requires_grad=False
        )
        self.sqrt_one_minus_alphas_cumprod = Tensor(
            (1. - alphas_cumprod) ** (1 / 2),
            device=device,
            requires_grad=False
        )

        self.posterior_variance = Tensor(
            self.betas.numpy() * (1. - self.alphas_cumprod_prev.numpy())
            / (1. - alphas_cumprod.numpy()),
            device=device,
            requires_grad=False
        )

    def q_sample(self, x_0, t, noise=None):
        '''
        q_sample - sample function in forward process
        Gets x_0 in range [-1, 1] as input
        '''
        shape = x_0.shape
        noise = x_0.device.randn(*shape) if noise is None else noise
        return (
            (extract(self.sqrt_alphas_cumprod, t, x_0.shape).broadcast_to(shape) * x_0 +
             extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape).broadcast_to(shape) * noise).data
        )

    def get_q_sample(self, x_0, t):
        '''
        Gets x_0 in range [-1, 1] as input
        '''
        t = Tensor([t], requires_grad=False)
        out = self.q_sample(x_0, t)

        out = (out + 1) / 2
        return out.data / out.numpy().max()

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = init.randn(*x_start.shape, device=x_start.device)
            if noise.shape != x_start.shape:
                noise = noise.reshape(x_start.shape)

        x_noisy = self.q_sample(x_0=x_start, t=t, noise=noise)
        predicted_noise = self.denoise_model(x_noisy, t)

        loss = self.loss_fn(predicted_noise, noise)

        return loss

    def p_sample(self, model, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape).data.broadcast_to(x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        ).data.broadcast_to(x.shape)
        sqrt_recip_alphas_t = extract(
            self.sqrt_recip_alphas, t, x.shape
        ).data.broadcast_to(x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = (sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )).data

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(
                self.posterior_variance, t, x.shape
            ).data.broadcast_to(x.shape)
            noise = init.randn(*x.shape, device=x.device, requires_grad=False)
            # Algorithm 2 line 4:
            return model_mean + ops.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)
    def p_sample_loop(self, shape):
        model = self.denoise_model
        device = model.parameters()[0].device

        batch_size = shape[0]
        # start from pure noise (for each example in the batch)
        img = init.randn(*shape, device=device, requires_grad=False)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)),
                      desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(
                model,
                img,
                init.constant(
                    batch_size,
                    c=i,
                    device=device,
                    requires_grad=False
                ),
                t_index=i
            )
            imgs.append(img.detach().numpy())
        return imgs

    def sample(self, image_size, batch_size, channels=3):
        return self.p_sample_loop(
            shape=(batch_size, channels, image_size, image_size))


def extract(x, t, x_shape):
    '''
    Same logics as a.gather(-1, t)
    '''
    batch_size = t.shape[0]
    device = x.device
    out_handle = device.empty((batch_size,))

    for i in range(batch_size):
        ind = int(t.numpy()[i])
        out_handle[i] = x.cached_data[ind]

    new_shape = (batch_size,) + (1,) * (len(x_shape) - 1)

    return Tensor(out_handle, device=device).reshape(new_shape)


def linear_beta_schedule(timesteps, device=None):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return Tensor(array_api.linspace(beta_start, beta_end, timesteps),
                  dtype="float32", device=device)


def cosine_beta_schedule(timesteps, s=0.008, device=None):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return Tensor(np.clip(betas, 0, 0.999), device=device, dtype="float32")

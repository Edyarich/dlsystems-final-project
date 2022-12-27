from typing import List, Optional
from needle.autograd import Tensor
from needle import ops, array_api
import needle.init as init
from needle.backend_ndarray.ndarray import BackendDevice
import numpy as np
from tqdm.auto import tqdm


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None,
                 dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(
                in_features,
                out_features,
                device=device,
                dtype=dtype
            )
        )
        self.has_bias = bias

        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(
                    out_features,
                    1,
                    device=device,
                    dtype=dtype
                ).transpose()
            )

    def forward(self, X: Tensor) -> Tensor:
        batch_size = X.shape[0]
        logits = X @ self.weight

        if self.has_bias:
            reshaped_bias = ops.broadcast_to(self.bias,
                                             (batch_size, self.out_features))
            logits += reshaped_bias

        return logits


class Flatten(Module):
    def forward(self, X):
        batch_size = X.shape[0]
        return X.reshape((batch_size, X.size // batch_size))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * Tensor(x.numpy() > 0, device=x.device)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        inp = x

        for module in self.modules:
            outp = module(inp)
            inp = outp

        return outp


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        num_classes = logits.shape[-1]

        log_sum_exp = ops.logsumexp(logits, -1)
        onehot_y = init.one_hot(
            num_classes, y, device=logits.device, dtype=logits.dtype
        )
        y_logits = ops.summation(logits * onehot_y, -1)

        return ops.mean(log_sum_exp - y_logits)


class L1Loss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor):
        return ops.abs(pred, target).mean()


class L2Loss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor):
        return ((target - pred) ** 2).mean()


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None,
                 dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        unsq_param_shape = [1] * len(x.shape)
        unsq_param_shape[-1] = self.dim
        weight = self.weight.reshape(unsq_param_shape).broadcast_to(x.shape)
        bias = self.bias.reshape(unsq_param_shape).broadcast_to(x.shape)

        if self.training:
            reduced_dims = 0,
            mean = ops.mean(x, reduced_dims, False)
            broad_mean = mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = ops.variation(x, reduced_dims, False)
            broad_var = var.reshape((1, self.dim)).broadcast_to(x.shape)
            normalized_x = (x - broad_mean) / ops.sqrt(broad_var + self.eps)

            self.running_mean *= (1 - self.momentum)
            self.running_mean += self.momentum * mean.data
            self.running_var *= (1 - self.momentum)
            self.running_var += self.momentum * var.data

            return weight * normalized_x + bias
        else:
            broad_mean = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            broad_var = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)
            normalized_x = (x - broad_mean) / ops.sqrt(broad_var + self.eps)

            return weight.data * normalized_x + bias.data


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # NCHW ==> NHWC
        s = x.shape
        _x = x.permute((0, 2, 3, 1)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.permute((0, 3, 1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        reduced_dims = tuple(range(1, len(x.shape)))
        mean = ops.mean(x, reduced_dims, True).broadcast_to(x.shape)
        var = ops.variation(x, reduced_dims, True).broadcast_to(x.shape)
        normalized_x = (x - mean) / ops.sqrt(var + self.eps)

        unsq_param_shape = [1] * len(x.shape)
        unsq_param_shape[-1] = self.dim
        weight = self.weight.reshape(unsq_param_shape).broadcast_to(x.shape)
        bias = self.bias.reshape(unsq_param_shape).broadcast_to(x.shape)

        return weight * normalized_x + bias


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p, device=x.device)
            return mask * x / (1 - self.p)
        else:
            return x


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: Optional[int] = None,
                 bias: int = True, device=None, dtype: str = "float32",
                 flip_kernel: bool = False):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2 if padding is None else padding

        self.flip_kernel = flip_kernel
        self.has_bias = bias

        self.weight = Parameter(
            init.kaiming_uniform(
                in_channels * kernel_size ** 2,
                out_channels * kernel_size ** 2,
                shape=(kernel_size, kernel_size, in_channels, out_channels),
                device=device,
                dtype=dtype
            )
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(
                    in_channels * kernel_size ** 2,
                    1,
                    shape=(out_channels,),
                    gain=3 ** -0.5,
                    device=device,
                    dtype=dtype
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        # NCHW ==> NHWC ==> NH'W'O ==> NOH'W'
        if self.flip_kernel:
            weight = ops.flip(self.weight, (0, 1))
        else:
            weight = self.weight

        _x = x.permute((0, 2, 3, 1))
        outp = ops.conv(_x, weight, self.stride, self.padding)

        if self.has_bias:
            outp += self.bias \
                .reshape((1, 1, 1, self.out_channels)) \
                .broadcast_to(outp.shape)

        return outp.permute((0, 3, 1, 2))


class ConvTranspose(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 bias: bool = True, device=None, dtype: str = "float32"):
        super().__init__()

        conv_padding = output_padding + kernel_size - 1 - padding
        self.conv = Conv(in_channels, out_channels, kernel_size, 1,
                         conv_padding, bias, device, dtype, True)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        dilated_x = ops.dilate(
            x, axes=(2, 3), dilation=self.stride - 1, cut_last=True
        )
        return self.conv(dilated_x)


class MaxPool(Module):
    """
    Multi-channel 2D MaxPool layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=0, stride=kernel_size
    No grouped convolution or dilation
    Only supports square kernels
    Image sizes should be divisible by kernel_size
    """

    def __init__(self, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        # NCHW ==> NHWC ==> NH'W'O ==> NOH'W'
        _x = x.permute((0, 2, 3, 1))
        output = ops.maxpool(_x, self.kernel_size)
        return output.permute((0, 3, 1, 2))


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + ops.exp(-x))


class Block(Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, device=None):
        super().__init__()
        self.time_mlp = Linear(time_emb_dim, out_ch, device=device)
        if up:
            self.conv1 = Conv(2 * in_ch, out_ch, 3, padding=1, device=device)
            self.transform = ConvTranspose(out_ch, out_ch, 4, 2, 1, device=device)
        else:
            self.conv1 = Conv(in_ch, out_ch, 3, padding=1, device=device)
            self.transform = MaxPool(2)
        self.conv2 = Conv(out_ch, out_ch, 3, padding=1, device=device)
        self.bnorm1 = BatchNorm2d(out_ch, device=device)
        self.bnorm2 = BatchNorm2d(out_ch, device=device)
        self.relu = ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.reshape(time_emb.shape + (1, 1)).broadcast_to(h.shape)

        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class Unet(Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self, device: Optional[BackendDevice] = None):
        super().__init__()
        image_channels = 3
        down_channels = (32, 64, 128, 256, 512)
        up_channels = (512, 256, 128, 64, 32)
        out_dim = 1
        time_emb_dim = 16

        # Time embedding
        self.time_mlp = Sequential(
            SinusoidalPosEmb(time_emb_dim),
            Linear(time_emb_dim, time_emb_dim, device=device),
            ReLU()
        )

        # Initial projection
        self.conv0 = Conv(image_channels, down_channels[0], 3, device=device)

        # Downsample
        self.downs = []
        # Upsample
        self.ups = []

        for i in range(len(down_channels) - 1):
            self.downs.append(
                Block(down_channels[i], down_channels[i + 1], time_emb_dim, device=device)
            )
            setattr(self, f'down_block_{i}', self.downs[-1])

        for i in range(len(up_channels) - 1):
            self.ups.append(
                Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True, device=device)
            )
            setattr(self, f'up_block_{i}', self.ups[-1])

        self.output = Conv(up_channels[-1], 3, out_dim, device=device)

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


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh',
                 device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        bound = 1 / hidden_size ** 0.5
        self.has_bias = bias

        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )

        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    1,
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )

            self.bias_hh = Parameter(
                init.rand(
                    1,
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype
                )
            )

        if nonlinearity == 'tanh':
            self.activation = Tanh()
        elif nonlinearity == 'relu':
            self.activation = ReLU()
        else:
            print(f"{nonlinearity} layer is not implemented")
            self.activation = Identity()

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        logits = X @ self.W_ih

        if h is not None:
            logits += h @ self.W_hh

        if self.has_bias:
            total_bias = self.bias_ih + self.bias_hh
            logits += total_bias.broadcast_to(logits.shape)

        return self.activation(logits)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()

        self.n_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cells = [
            RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]

        for _ in range(num_layers - 1):
            self.rnn_cells.append(
                RNNCell(hidden_size, hidden_size, bias, nonlinearity, device,
                        dtype)
            )

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape
        if h0 is None:
            h0 = init.zeros(self.n_layers, bs, self.hidden_size,
                            device=X.device,
                            requires_grad=True)

        h0_arr = [x for x in ops.split(h0, 0)]

        X = ops.split(X, 0)
        output = []

        for i in range(seq_len):
            for j in range(self.n_layers):
                inp = X[i] if j == 0 else h0_arr[j - 1]
                h0_arr[j] = self.rnn_cells[j](inp, h0_arr[j])

                if j + 1 == self.n_layers:
                    output.append(h0_arr[j])

        return ops.stack(output, 0), ops.stack(h0_arr, 0)


class LSTMCell(Module):
    K_GATES = 4

    def __init__(self, input_size, hidden_size, bias=True, device=None,
                 dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        self.has_bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size

        bound = 1 / hidden_size ** 0.5

        self.W_ih = Parameter(
            init.rand(
                input_size,
                self.K_GATES * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                self.K_GATES * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    1,
                    self.K_GATES * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    1,
                    self.K_GATES * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype
                )
            )

        self.activations = (Sigmoid(), Sigmoid(), Tanh(), Sigmoid())
        self.tanh = Tanh()

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        logits = X @ self.W_ih
        bs = X.shape[0]

        if h is not None:
            h0, c0 = h
            logits += h0 @ self.W_hh
        else:
            h0 = init.zeros(bs, self.hidden_size, device=X.device,
                            dtype=X.dtype,
                            requires_grad=True)
            c0 = init.zeros(bs, self.hidden_size, device=X.device,
                            dtype=X.dtype,
                            requires_grad=True)

        if self.has_bias:
            total_bias = self.bias_ih + self.bias_hh
            logits += total_bias.broadcast_to(logits.shape)

        gates = ops.split(logits.reshape((bs, self.K_GATES, self.hidden_size)),
                          axis=1)
        gates = [activ(gate) for activ, gate in zip(self.activations, gates)]
        inp_gate, fgt_gate, cell_inp, outp_gate = gates

        new_cell = fgt_gate * c0 + inp_gate * cell_inp
        new_hidden = outp_gate * self.tanh(new_cell)

        return new_hidden, new_cell


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.n_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cells = [
            LSTMCell(input_size, hidden_size, bias, device, dtype)]

        for _ in range(num_layers - 1):
            self.lstm_cells.append(
                LSTMCell(hidden_size, hidden_size, bias, device, dtype)
            )

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape
        if h is None:
            h0 = init.zeros(self.n_layers, bs, self.hidden_size,
                            device=X.device,
                            requires_grad=True)
            c0 = init.zeros(self.n_layers, bs, self.hidden_size,
                            device=X.device,
                            requires_grad=True)
        else:
            h0, c0 = h

        h0_arr = [x for x in ops.split(h0, 0)]
        c0_arr = [x for x in ops.split(c0, 0)]

        X = ops.split(X, 0)
        output = []

        for i in range(seq_len):
            for j in range(self.n_layers):
                inp = X[i] if j == 0 else h0_arr[j - 1]
                hid = (h0_arr[j], c0_arr[j])
                h0_arr[j], c0_arr[j] = self.lstm_cells[j](inp, hid)

                if j + 1 == self.n_layers:
                    output.append(h0_arr[j])

        return ops.stack(output, 0), (
        ops.stack(h0_arr, 0), ops.stack(c0_arr, 0))


class SinusoidalPosEmb(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        # x.shape = (batch_size,)
        # Returns emb with shape = (batch_size, self.dim)
        import math
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = ops.exp(Tensor(range(half_dim), device=device) * -emb)
        emb = x.broadcast_to((x.size, 1)) @ emb.broadcast_to((1, emb.size))
        emb = ops.stack(
            (ops.sin(emb), ops.cos(emb)), axis=2
        ).reshape((x.size, self.dim))
        return emb


class Diffusion(Module):
    def __init__(
        self,
        model,
        optimizer,
        timesteps,
        beta_schedule="linear",
        loss_type="l1",
        device=None
    ):
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps, device=device)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps, device=device)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.model = model
        self.optimizer = optimizer
        if loss_type == "l1":
            self.loss_fn = L1Loss()
        elif loss_type == "l2":
            self.loss_fn = L2Loss()
        else:
            raise NotImplementedError(f"Unknown loss {loss_type}")

        alphas = 1.0 - betas
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
            betas.numpy() * (1. - self.alphas_cumprod_prev.numpy())
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
        denoise_model = self.model
        if noise is None:
            noise = init.randn(*x_start.shape, device=x_start.device)
            if noise.shape != x_start.shape:
                noise = noise.reshape(x_start.shape)

        x_noisy = self.q_sample(x_0=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        loss = self.loss_fn(noise, predicted_noise)

        return loss

    def p_sample(self, model, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(
            self.sqrt_recip_alphas, t, x.shape
        )

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
            )
            noise = init.randn(*x.shape, device=x.device, requires_grad=False)
            # Algorithm 2 line 4:
            return model_mean + ops.sqrt(posterior_variance_t).data * noise

    # Algorithm 2 (including returning all images)
    def p_sample_loop(self, shape):
        model = self.model
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = init.randn(*shape, device=device, requires_grad=False)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)),
                      desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img,
                                init.constant((b,), i, device=device,
                                              dtype="int64",
                                              requires_grad=False), i)
            imgs.append(img.cpu().numpy())
        return imgs

    def sample(self, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(
            shape=(batch_size, channels, image_size, image_size))

    def forward(self, X: Tensor) -> Tensor:
        self.sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[self.t]
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[
            self.t]

        noise = init.randn(X.shape, device=X.device)
        return self.sqrt_alpha_cumprod * X + self.sqrt_one_minus_alpha_cumprod * noise, noise


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


def normalize_minus_one_to_one(img):
    return img * 2 - 1


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None,
                 dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

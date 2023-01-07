import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from apps.diffusion import Unet, Diffusion

sys.path.append('./python')

import needle as ndl
from needle import array_api, Tensor, NDArray

DATA_DIR = Path('./data/landscapes/')
IMAGE_FILES = sorted(str(x) for x in list(DATA_DIR.rglob('*.jpg')))


if __name__ == '__main__':
    transforms = [
        ndl.data.RandomFlipHorizontal(),
        ndl.data.Lambda(lambda img: np.transpose(img, (2, 0, 1))),
    ]
    dataset = ndl.data.LandscapesDataset(
        IMAGE_FILES, extra_transforms=transforms, img_size=64
    )

    device = ndl.cpu()
    model = Unet(device=device)
    optimizer = ndl.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 1
    batch_size = 12
    dataloader = ndl.data.DataLoader(dataset, batch_size)

    timesteps = 50
    module = Diffusion(model, timesteps, loss_type="l2",
                          device=device)

    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            optimizer.reset_grad()
            batch = ndl.Tensor(batch, device=device)
            ts = Tensor(np.random.randint(0, timesteps, (len(batch),)),
                       device=device,
                       requires_grad=False)

            loss = module.p_losses(batch, ts)

            if step % 10 == 0:
                print("Loss:", loss.cached_data[0])

            loss.backward()
            optimizer.step()

    result = module.sample(64, 4, 3)
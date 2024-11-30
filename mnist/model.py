from flax import nnx
from functools import partial
from jax import numpy as jnp

class CNN(nnx.Module):
    def __init__(self, *, rngs:nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size = (3,3), rngs = rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size = (3,3), rngs = rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape = (2,2), strides = (2,2))
        self.linear1 = nnx.Linear(3136, 256, rngs = rngs)
        self.linear2 = nnx.Linear(256, 10, rngs = rngs)

    def __call__(self,x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape((x.shape[0], -1))
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    from dataset import get_loaders
    train_loader, valid_loader = get_loaders()
    model = CNN(rngs = nnx.Rngs(0))
    for batch in train_loader:
        image = batch['image']
        response = model(image)
        print(response.shape)
        break

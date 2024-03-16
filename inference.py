import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from config import InferenceConfig
from src.data import InferenceDataset, transformation
from src.data.visualisation import batch_to_numpy
from src.models import GAN

cfg = InferenceConfig.load("train")

model = GAN(
    dtype=cfg.data.dtype,
    input_channels=cfg.data.input_channels,
    output_channels=cfg.data.output_channels,
    generator_kwargs=cfg.network.generator.model_dump(),
    discriminator_kwargs=cfg.network.discriminator.model_dump(),
)

model.load_state_dict(torch.load(cfg.inference.weights_path))
model.eval()

input_size = cfg.network.generator.input_size
transform = Compose(
    [
        transformation.Resize((input_size, input_size), antialias=True),
        transformation.To(torch.device("cuda:0")),
    ]
)

dataset = InferenceDataset(cfg.inference.in_data_path)
dataloader = DataLoader(dataset, batch_size=2)

sample_counter = 0
for batch in dataloader:
    batch = transform(batch)
    pred = model(batch)
    imgs_inference = batch_to_numpy(pred)

    for img_inference in imgs_inference:
        plt.imsave(
            f"data/inference/pred/{sample_counter}.png",
            (img_inference * 255).astype(np.uint8),
        )
        sample_counter += 1

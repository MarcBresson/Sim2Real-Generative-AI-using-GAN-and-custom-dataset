from pathlib import Path
import yaml

import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from src.data import InferenceDataset
from src.data import transformation
from src.models import GAN


def load_config(config_name: str, dir_: Path = Path("config")):
    filepath = dir_ / (config_name + ".yml")

    with filepath.open() as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


cfg = load_config("train", Path("config"))

model = GAN(
    dtype=cfg["data"]["dtype"],
    input_channels=cfg["data"]["input_channels"],
    output_channels=cfg["data"]["output_channels"],
    generator_kwargs=cfg["network"]["generator"],
    discriminator_kwargs=cfg["network"]["discriminator"]
)

model.eval()

transform = Compose([
    transformation.To(torch.device("cuda:0")),
    transformation.Resize((256, 256), antialias=True),
])

dataset = InferenceDataset(Path("data", "inference"))
dataloader = DataLoader(dataset, batch_size=2)

for batch in dataset:
    pred = model(batch)


# inference

from pathlib import Path
from typing import Annotated, overload, TypeAlias
import yaml

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, FilePath, DirectoryPath
from pydantic.functional_validators import AfterValidator


def new_path_and_parents(path: Path) -> Path:
    assert not path.exists(), f"{path} already exists"
    return path


NewPathAndParents: TypeAlias = Annotated[Path, AfterValidator(new_path_and_parents)]


class Data(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_channels: int
    output_channels: int
    dtype: str


class OptimizerKwargs(BaseModel):
    lr: float = 1e-4
    betas: tuple[float, float] = (0.0, 0.999)


class Generator(BaseModel):
    num_upsampling_layers: str
    n_filters: int
    input_size: int
    lr: float
    optimizer_kwargs: OptimizerKwargs


class Discriminator(BaseModel):
    n_layers: int
    n_filters: int
    lr: float
    optimizer_kwargs: OptimizerKwargs


class Network(BaseModel):
    generator: Generator
    discriminator: Discriminator


class Train(BaseModel):
    class Checkpointer(BaseModel):
        period: int

    class Visualisation(BaseModel):
        n_samples: int
        every_nth_epoch: int

    in_data_path: DirectoryPath
    out_data_path: NewPathAndParents
    n_epochs: int
    batch_size: int
    dataset_split: list[float | int]
    profile: bool = False
    checkpointer: Checkpointer
    visualisation: Visualisation


class Inference(BaseModel):
    in_data_path: DirectoryPath
    out_data_path: NewPathAndParents
    weights_path: FilePath


class Config(BaseModel):
    data: Data
    network: Network
    train: Train | None = None
    inference: Inference | None = None

    @overload
    @classmethod
    def load(cls, *, path: Path) -> Self: ...

    @overload
    @classmethod
    def load(cls, config_name: str, *, directory: Path = Path("config")) -> Self: ...

    @classmethod
    def load(
        cls,
        config_name: str | None = None,
        directory: Path = Path("config"),
        path: Path | None = None,
    ) -> Self:
        """
        load a config into the pydantic model

        Parameters
        ----------
        config_name : str | None, optional
            Will load the config `config_name.yml` in the directory
            `directory`, by default None
        directory : Path, optional
            Directory in which to look for the `.yml` config file, by
            default Path("config")
        path : Path | None, optional
            If specified, will load the config file at this path, by
            default None

        Returns
        -------
        Self
        """
        if path is None:
            if config_name is None:
                raise ValueError("either path or config_name must be speficied")

            path = directory / (config_name + ".yml")

        with path.open() as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
            return cls.model_validate(cfg)


class TrainConfig(Config):
    train: Train


class InferenceConfig(Config):
    inference: Inference

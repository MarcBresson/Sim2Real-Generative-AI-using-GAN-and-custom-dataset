from typing import Any

from pydantic import BaseModel

from config.base_models import *  # noqa: F403


def get_config_to_dict(conf: BaseModel | None) -> dict[str, Any]:
    """convert a configuration to a dictionnary ready to be unpacked."""
    if conf is None:
        return {}

    if isinstance(conf, BaseModel):
        return dict(conf)

    return conf

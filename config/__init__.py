from typing import Any

from pydantic import BaseModel

from config.base_models import *


def model_dump_first_level(model: BaseModel) -> dict[str, Any | BaseModel]:
    """
    Dump only the first level of a pydantic level.
    All fields that are BaseModel will stay untouched.
    """
    basemodel_fields = set()

    for field, v in model:
        if isinstance(v, BaseModel):
            basemodel_fields.add(field)

    pythonized_fields = model.model_dump(exclude=basemodel_fields)

    for field in basemodel_fields:
        pythonized_fields[field] = getattr(model, field)

    return pythonized_fields

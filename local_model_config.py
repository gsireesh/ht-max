from dataclasses import dataclass
from typing import Callable

from papermage.predictors import BasePredictor
from streamlit import cache_resource

from papermage_components.table_transformer_structure_predictor import (
    TableTransformerStructurePredictor,
)


@dataclass
class LocalModelInfo:
    model_name: str
    model_desc: str
    get_model: Callable[[], BasePredictor]


@cache_resource
def get_table_transformer_predictor():
    return TableTransformerStructurePredictor.from_model_name()


MODEL_LIST: list[LocalModelInfo] = [
    LocalModelInfo(
        model_name="Table Transformer Structure Parser",
        model_desc="A model to parse structure from table images.",
        get_model=get_table_transformer_predictor,
    ),
]

AVAILABLE_LOCAL_MODELS = {m.model_name: m for m in MODEL_LIST}

from dataclasses import dataclass
from typing import Callable

from papermage.predictors import BasePredictor
from streamlit import cache_resource

from papermage_components.table_transformer_structure_predictor import (
    TableTransformerStructurePredictor,
)
from papermage_components.chem_data_extractor_predictor import (
    ChemDataExtractorPredictor,
)


@dataclass
class LocalModelInfo:
    model_name: str
    model_desc: str
    get_model: Callable[[], BasePredictor]


@cache_resource
def get_table_transformer_predictor():
    return TableTransformerStructurePredictor.from_model_name()


def get_cde_predictor():
    return ChemDataExtractorPredictor(cde_service_url="http://panther.lti.cs.cmu.edu:8001")


MODEL_LIST: list[LocalModelInfo] = [
    LocalModelInfo(
        model_name="Table Transformer Structure Parser",
        model_desc="A model to parse structure from table images.",
        get_model=get_table_transformer_predictor,
    ),
    LocalModelInfo(
        model_name="ChemDataExtractor Token Tagger",
        model_desc="A model that uses ChemDataExtractor to tag chemicals.",
        get_model=get_cde_predictor,
    ),
]

AVAILABLE_LOCAL_MODELS = {m.model_name: m for m in MODEL_LIST}

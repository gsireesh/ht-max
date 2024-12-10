from dataclasses import dataclass
import os
from typing import Callable

from papermage.predictors import BasePredictor
from streamlit import cache_resource

from app_config import app_config as config
from papermage_components.matie_service_predictor import MatIEServicePredictor
from papermage_components.table_transformer_structure_predictor import (
    TableTransformerStructurePredictor,
)
from papermage_components.chem_data_extractor_predictor import (
    ChemDataExtractorPredictor,
)
from papermage_components.table_structure_predictor_mathpix import MathPixTableStructurePredictor


@dataclass
class LocalModelInfo:
    model_name: str
    model_desc: str
    get_model: Callable[[], BasePredictor]
    use_by_default: bool


@cache_resource
def get_table_transformer_predictor():
    return TableTransformerStructurePredictor.from_model_name()


def get_cde_predictor():
    return ChemDataExtractorPredictor(cde_service_url=config["chemdataextractor_service_url"])


def get_matie_predictor():
    return MatIEServicePredictor(matIE_service_url=config["matie_service_url"])


def get_mathpix_predictor():
    if not config["mathpix_credentials"] or not config["mathpix_credentials"]["app_key"]:
        raise AssertionError("No MathPix API Key provided in config! Skipping predictor.")

    return MathPixTableStructurePredictor(mathpix_headers=config["mathpix_credentials"])


MODEL_LIST: list[LocalModelInfo] = [
    LocalModelInfo(
        model_name="Table Transformer Structure Parser",
        model_desc="A model to parse structure from table images.",
        get_model=get_table_transformer_predictor,
        use_by_default=True,
    ),
    LocalModelInfo(
        model_name="MathPix Table Structure Transformer",
        model_desc="Parse tables with MathPix",
        get_model=get_mathpix_predictor,
        use_by_default=False,
    ),
    LocalModelInfo(
        model_name="ChemDataExtractor Token Tagger",
        model_desc="A model that uses ChemDataExtractor to tag chemicals.",
        get_model=get_cde_predictor,
        use_by_default=False,
    ),
    LocalModelInfo(
        model_name="MatIE Information Extractor",
        model_desc="A model that tags material properties and relations.",
        get_model=get_matie_predictor,
        use_by_default=True,
    ),
]

AVAILABLE_LOCAL_MODELS = {m.model_name: m for m in MODEL_LIST}

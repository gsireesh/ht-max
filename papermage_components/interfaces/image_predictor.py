from abc import ABC
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import pandas as pd
from papermage import Document, Entity, Metadata
from papermage.predictors import BasePredictor
from tqdm.auto import tqdm

from papermage_components.utils import get_table_image


@dataclass
class ImagePredictionResult:
    raw_prediction: dict
    predicted_table: pd.DataFrame = None
    predicted_text: str = None
    predicted_dict: dict = None


class ImagePredictor(BasePredictor, ABC):
    def __init__(self, entity_to_process: str):
        self.entity_to_process = entity_to_process

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [self.entity_to_process]

    @property
    def predictor_identifier(self) -> str:
        raise NotImplementedError

    @property
    def preferred_layer_name(self) -> str:
        raise NotImplementedError

    def process_image(self, image) -> ImagePredictionResult:
        raise NotImplementedError

    def _predict(self, doc: Document) -> list[Entity]:
        all_entities = []

        for entity in tqdm(getattr(doc, self.entity_to_process)):
            entity_image = get_table_image(entity, doc)
            predicted_result = self.process_image(entity_image)

            meta_dict = {k: v for k, v in asdict(predicted_result) if v is not None}
            image_entity = Entity(
                spans=entity.spans,
                boxes=entity.boxes,
                images=entity.images,
                metadata=Metadata(**meta_dict),
            )
            all_entities.append(image_entity)

        return all_entities

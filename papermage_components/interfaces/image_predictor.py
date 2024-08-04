from abc import ABC
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import pandas as pd
from papermage import Box, Document, Entity, Metadata, CaptionsFieldName
from papermage.predictors import BasePredictor
from tqdm.auto import tqdm

from papermage_components.utils import get_table_image, globalize_box_coordinates


def get_nearby_captions(table, doc, expansion_factor):
    box = table.boxes[0]

    exp_h = expansion_factor * box.h
    diff_h = exp_h - box.h

    search_box = Box(l=box.l, t=box.t - diff_h / 2, w=box.w, h=exp_h, page=box.page)
    potential_captions = doc.find(query=search_box, name=CaptionsFieldName)
    return potential_captions


@dataclass
class ImagePredictionResult:
    raw_prediction: dict

    # if the predictions take the form of either a table, or arbitrary key/value.
    predicted_table: pd.DataFrame = None
    predicted_dict: dict = None

    # if the predictions are bounding boxes.
    predicted_boxes: list[Box] = None

    # predicted text is for e.g. image captioning models
    predicted_text: str = None


class ImagePredictorABC(BasePredictor, ABC):
    def __init__(self, entity_to_process: str, find_caption: bool = True):
        self.entity_to_process = entity_to_process
        self.find_caption = find_caption

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [self.entity_to_process] + ([CaptionsFieldName] if self.find_caption else [])

    @property
    def predictor_identifier(self) -> str:
        raise NotImplementedError

    @property
    def preferred_layer_name(self) -> str:
        raise NotImplementedError

    def process_image(self, image) -> ImagePredictionResult:
        raise NotImplementedError

    def process_entity(self, entity) -> ImagePredictionResult:
        entity_image = get_table_image(entity, entity.layer.doc)
        return self.process_image(entity_image)

    def _predict(self, doc: Document) -> list[Entity]:
        all_entities = []

        for entity in tqdm(getattr(doc, self.entity_to_process)):
            try:
                if len(entity.boxes) > 1:
                    raise AssertionError("Entity has more than one box!")

                predicted_result = self.process_entity(entity)

                meta_dict = {k: v for k, v in asdict(predicted_result).items() if v is not None}

                if "predicted_boxes" in meta_dict:
                    meta_dict["predicted_boxes"] = [
                        globalize_box_coordinates(box, entity.boxes[0], doc).to_json()
                        for box in meta_dict["predicted_boxes"]
                    ]

            except Exception as e:
                print("Failed to parse table! Continuing")
                continue

            if self.find_caption:
                predicted_caption = "No caption found."
                candidate_table_captions = get_nearby_captions(entity, doc, expansion_factor=1.4)
                if candidate_table_captions:
                    if len(candidate_table_captions) > 1:
                        best_candidate = None
                        min_dist = 1
                        for caption in candidate_table_captions:
                            if abs(caption.boxes[0].t - entity.boxes[0].t) < min_dist:
                                best_candidate = caption
                    else:
                        best_candidate = candidate_table_captions[0]
                    predicted_caption = best_candidate.text
                meta_dict["predicted_caption"] = predicted_caption

            image_entity = Entity(
                spans=entity.spans,
                boxes=entity.boxes,
                images=entity.images,
                metadata=Metadata(**meta_dict),
            )
            all_entities.append(image_entity)

        return all_entities

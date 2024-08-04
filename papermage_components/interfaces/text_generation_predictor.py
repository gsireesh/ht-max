from abc import ABC
from dataclasses import dataclass
import json
from json import JSONDecodeError
from typing import Callable, List, Optional

from tqdm.auto import tqdm

from papermage import Document, Entity, Metadata
from papermage.predictors import BasePredictor


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMValidationResult:
    is_valid: bool
    failure_message: str


@dataclass
class LLMResults:
    predicted_text: str
    results_table: Optional[dict] = None


def get_prompt_generator(prompt_text: str) -> Callable[[str], List[LLMMessage]]:
    return lambda text: [LLMMessage(role="user", content=f"{prompt_text}\n\n{text}")]


class TextGenerationPredictorABC(BasePredictor, ABC):
    def __init__(self, entity_to_process):
        self.entity_to_process = entity_to_process

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [self.entity_to_process]

    def validate(self):
        return True

    @property
    def predictor_identifier(self) -> str:
        raise NotImplementedError

    @property
    def preferred_layer_name(self):
        return f"TAGGED_GENERATION_{self.predictor_identifier}"

    def generate_from_entity_text(self, entity_text: str) -> str:
        raise NotImplementedError

    def postprocess_text_to_dict(self, text) -> Optional[dict]:
        try:
            return json.loads(text)
        except JSONDecodeError:
            print("Failed to parse JSON!")
            return None

    def _predict(self, doc: Document) -> list[Entity]:
        all_entities = []

        for entity in tqdm(getattr(doc, self.entity_to_process)):
            generated_text = self.generate_from_entity_text(entity.text)
            parsed_table = self.postprocess_text_to_dict(generated_text)
            predicted_entity = Entity(
                spans=entity.spans,
                boxes=entity.boxes,
                images=entity.images,
                metadata=Metadata(predicted_text=generated_text, predicted_table=parsed_table),
            )
            all_entities.append(predicted_entity)
        return all_entities

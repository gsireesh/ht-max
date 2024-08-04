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
    """
    Dataclass representing a role and message, in the OpenAI dialogue format.
    """

    role: str
    content: str


@dataclass
class LLMValidationResult:
    """
    Class to represent if the LLM predictor is valid, i.e. can make requests.
    """

    is_valid: bool
    failure_message: str


@dataclass
class LLMResults:
    """
    Union class for both predicted text and a parsed table for the LLM predictor.

    predicted_text represents the raw predicted text, and results_table is a dict that can have
    pd.DataFrame() called on it to render it nicely.
    """

    predicted_text: str
    results_table: Optional[dict] = None


def get_prompt_generator(prompt_text: str) -> Callable[[str], List[LLMMessage]]:
    """Returns a function that takes a string and customizes a set of user/system messages to it.

    Parameters
    ----------
    prompt_text : The text of the prompt.

    Returns
    -------
    A function that, when called with a string, returns a list of system and user messages to prompt
    an LLM.

    """
    return lambda text: [LLMMessage(role="user", content=f"{prompt_text}\n\n{text}")]


class TextGenerationPredictorABC(BasePredictor, ABC):
    """
    Interface to implement for text-to-text generation in collage.
    """

    def __init__(self, entity_to_process):
        """
        Parameters
        ----------
        entity_to_process : represents the PaperMage layer whose entities will be iterated through.
        """
        self.entity_to_process = entity_to_process

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [self.entity_to_process]

    def validate(self):
        return True

    @property
    def predictor_identifier(self) -> str:
        """MUST IMPLEMENT THIS! Typically, just the name of the underlying model."""
        raise NotImplementedError

    @property
    def preferred_layer_name(self):
        return f"TAGGED_GENERATION_{self.predictor_identifier}"

    def generate_from_entity_text(self, entity_text: str) -> str:
        """MUST IMPLEMENT THIS! Implement the text-to-text function."""
        raise NotImplementedError

    def postprocess_text_to_dict(self, text) -> Optional[dict]:
        """Optional Implementation: if you'd like to do more involved processing on your LLM results."""
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

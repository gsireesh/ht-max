from dataclasses import asdict, dataclass
import json
from json import JSONDecodeError
from typing import Callable, List, Optional

from litellm import (
    completion,
    check_valid_key,
    validate_environment,
    open_ai_text_completion_models,
    anthropic_models,
)
from papermage import Entity, Document, Metadata
from papermage.predictors import BasePredictor

from papermage_components.interfaces.text_generation_predictor import *


AVAILABLE_LLMS = open_ai_text_completion_models + anthropic_models


DEFAULT_MATERIALS_PROMPT = """I am working on identifying various entities related to materials science within texts. Below are the categories of entities I'm interested in, along with their definitions and examples. Please read the input text and identify entities according to these categories:
Material: Main materials system discussed/developed/manipulated or material used for comparison. Example: Nickel-based Superalloy.
Participating Materials: Anything interacting with the main material by addition, removal, or as a reaction catalyst. Example: Zirconium.
Synthesis: Process/tools used to synthesize the material. Examples: Laser Powder Bed Fusion (specific), alloy development (vague).
Characterization: Tools used to observe and quantify material attributes (e.g., microstructure features, chemical composition, mechanical properties). Examples: X-ray Diffraction, EBSD, creep test.
Environment: Describes the synthesis/characterization/operation conditions/parameters used. Examples: Temperature (specific), applied stress, welding conditions (vague).
Phenomenon: Something that is changing (either on its own or as a direct/indirect result of an operation) or observable. Examples: Grain boundary sliding (specific), (stray grains) formation, (GB) deformation (vague).
MStructure: Location-specific features of a material system on the "meso"/"macro" scale. Examples: Drainage pathways (specific), intersection (between the nodes and ligaments) (vague).
Microstructure: Location-specific features of a material system on the "micro" scale. Examples: Stray grains (specific), GB, slip systems.
Phase: Materials phase (atomic scale). Example: Gamma precipitate.
Property: Any material attribute. Examples: Crystallographic orientation, GB character, environment resistance (mostly specific).
Descriptor: Indicates some description of an entity. Examples: High-angle boundary, (EBSD) maps, (nitrogen) ions.
Operation: Any (non/tangible) process/action that brings change in an entity. Examples: Adding/increasing (Co), substituted, investigate.
Result: Outcome of an operation, synthesis, or some other entity. Examples: Greater retention, repair (defects), improve (part quality).
Application: Final-use state of a material after synthesis/operation(s). Example: Thermal barrier coating.
Number: Any numerical value within the text.
Amount Unit: Unit of the number
For each identified entity, please provide the entity text, the category from the schema above, and the context in which the entity was identified.
Format your output as below:
[
    {{
    "entity": "entity1 name",
    "category": "entity2 category",
    "context": "entity2 context",
    }}
    {{
    "entity": "entity2 name",
    "category": "entity2 category",
    "context": "entity2 context",
    }}
]


Recognize entities in the following text:
        {text}
"""


class LiteLlmCompletionPredictor(TextGenerationPredictorABC):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        prompt_generator_function: Callable[[str], List[LLMMessage]],
        entity_to_process="reading_order_sections",
    ):
        super().__init__(entity_to_process)
        self.model_name = model_name
        self.api_key = api_key
        self.generate_prompt = prompt_generator_function

    def validate(self):
        env_validation = validate_environment(model=self.model_name, api_key=self.api_key)
        if missing_keys := env_validation["missing_keys"]:
            return LLMValidationResult(False, f"Missing credentials: {missing_keys}")
        elif not check_valid_key(model=self.model_name, api_key=self.api_key):
            return LLMValidationResult(False, "Invalid API Key!")
        else:
            return LLMValidationResult(True, "")

    @property
    def predictor_identifier(self) -> str:
        return self.model_name

    @property
    def preferred_layer_name(self):
        return f"TAGGED_GENERATION_{self.predictor_identifier}"

    def generate_from_entity_text(self, entity_text: str) -> str:
        messages = [asdict(m) for m in self.generate_prompt(entity_text)]
        llm_response = completion(
            model=self.model_name, api_key=self.api_key, messages=messages, max_tokens=2500
        )
        response_text = llm_response.choices[0].message.content
        return response_text

from dataclasses import dataclass
from typing import Callable, List

from litellm import completion, check_valid_key, validate_environment
from papermage import Entity, Document, Metadata
from papermage.predictors import BasePredictor


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMValidationResult:
    is_valid: bool
    failure_message: str


def generate_materials_ie_prompt(text: str) -> List[LLMMessage]:
    default_message = """I am working on identifying various entities related to materials science within texts. Below are the categories of entities I'm interested in, along with their definitions and examples. Please read the input text and identify entities according to these categories:
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
{{
"entities": [
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
}}


Recognize entities in the following text:
        {text}
"""

    return [LLMMessage(role="user", content=default_message.format(text=text))]


class LLMCompletionPredictor(BasePredictor):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        prompt_generator_function: Callable[[str], List[LLMMessage]],
        entity_to_process="reading_order_sections",
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.generate_prompt = prompt_generator_function
        self.entity_to_process = entity_to_process

    def validate(self):
        env_validation = validate_environment(model=self.model_name, api_key=self.api_key)
        if missing_keys := env_validation["missing_keys"]:
            return LLMValidationResult(False, f"Missing credentials: {missing_keys}")
        elif not check_valid_key(model=self.model_name, api_key=self.api_key):
            return LLMValidationResult(False, "Invalid API Key!")
        else:
            return LLMValidationResult(True, "")

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [self.entity_to_process]

    @property
    def predictor_identifier(self) -> str:
        return self.model_name

    @property
    def preferred_layer_name(self):
        return f"TAGGED_GENERATION_{self.predictor_identifier}"

    def generate_from_entity(self, entity: Entity) -> str:
        messages = self.generate_prompt(entity.text)
        llm_response = completion(model=self.model_name, api_key=self.api_key, messages=messages)
        response_text = llm_response.choices[0].message.content
        return response_text

    def _predict(self, doc: Document) -> list[Entity]:
        all_entities = []

        for entity in getattr(doc, self.entity_to_process):
            generated_text = self.generate_from_entity(entity)
            predicted_entity = Entity(
                spans=entity.spans,
                boxes=entity.boxes,
                images=entity.images,
                metadata=Metadata(predicted_text=generated_text),
            )
            all_entities.append(predicted_entity)
        return all_entities

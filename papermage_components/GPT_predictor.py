import json
from json import JSONDecodeError
from typing import List, Tuple

import openai
from papermage.magelib import (
    Document,
    Prediction,
)
from papermage.predictors import BasePredictor
from tqdm.auto import tqdm


class GPT_predictor(BasePredictor):
    def __init__(
        self,
        api_key="",
        temperature=0.7,
        max_tokens=1000,
    ):
        # Set your OpenAI API key
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return ["reading_order_sections"]

    def _predict(self, doc: Document) -> Tuple[Prediction, ...]:
        for paragraph in tqdm(doc.reading_order_sections):
            if len(paragraph.text) < 100:
                continue
            gpt_result = self.run_gpt_ner(paragraph.text)
            try:
                gpt_entities_dict = json.loads(gpt_result)
            except JSONDecodeError as e:
                continue
            all_gpt_entities = []
            for entity in gpt_entities_dict["entities"]:
                e_type = entity["category"]
                all_gpt_entities.append(
                    {
                        "entity_string": entity["entity"],
                        "entity_type": e_type,
                        "entity_context": entity["context"],
                    }
                )
            paragraph.metadata["gpt_recognized_entities"] = all_gpt_entities

        return tuple()

    def run_gpt_ner(self, article):

        openai.api_key = self.api_key

        Schema_And_Prompt = """I am working on identifying various entities related to materials science within texts. Below are the categories of entities I'm interested in, along with their definitions and examples. Please read the input text and identify entities according to these categories:
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
{
"entities": [
    {
    "entity": "entity1 name",
    "category": "entity2 category",
    "context": "entity2 context",
    }
    {
    "entity": "entity2 name",
    "category": "entity2 category",
    "context": "entity2 context",
    }
]
}


Recognize entities in the following text:
        
"""
        # Parameters for the completion
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",  # free version of GPT model
            prompt=Schema_And_Prompt + article,
            max_tokens=self.max_tokens,  # Adjust max tokens according to your needs
            temperature=self.temperature,  # Adjust temperature according to your needs
            n=1,  # Number of completions to generate
            stop=None,  # Custom stop sequence to end the completion
        )

        # Get the generated text from the response
        result = response.choices[0].text.strip()
        # print('result\n',result)
        return result

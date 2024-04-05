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
        max_tokens=2500,
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

        Schema_And_Prompt = """| Entity    | Definition          | Examples            |
|-----------|---------------------|---------------------|
| Material   | main materials system discussed / developed / manipulated OR material used for comparison      | Nickel-based Superalloy      |
|  Participating Materials         | anything interacting with the main material by addition, removal or as a reaction catalyst| Zirconium          |
|    Synthesis |  process/tools used to synthesize the material       |  Laser Powder Bed Fusion (specific), alloy development (vague)     |
| Characterization | tools used to observe and quantify material at- tributes (e.g., microstructure features, chemical composition, mechanical properties, etc.)        | X-ray Diffraction, EBSD, creep test     |
|   Environment        | describes the synthesis / characterization / op- eration – conditions / parameters used    | temperature (specific), applied stress, welding conditions (vague)   |
|  Phenomenon         | something that is changing (either on its own or as an direct/indirect result of an operation) or observable        | grain boundary sliding (specific), (stray grains) formation, (GB) deformation (vague)                    |
|MStructure|location specific features of a material system on the "meso" / "macro" scale|drainage pathways (specific), inter- section (between the nodes and liga- ments) (vague)|
| Microstructure      | location specific features of a material system on the "micro" scale   | stray grains (specific), GB, slip systems              |
|     Phase      | materials phase (atomic scale)           | gamma precipitate     |
|     Property   | any material attribute           | crystallographic orientation, GB chacacter, environment resistance (mostly specific)     |
|     Descriptor      | indicates some description of an entity           | high-angle boundary, (EBSD) maps, (nitrogen) ions     |
|     Operation      |  any (non/tangible) process / action that brings change in an entity          | adding / increasing (Co), substituted, investigate     |
|     Result     | outcome of an operation, synthesis, or some other entity           | greater retention, repair (defects), improve (part quality)     |
|     Application      | final-use state of a material after synthesis / operation(s)          | thermal barrier coating     |
|     Number      | any numerical value within the text          |      |
|     Amount Unit      | unit of the number           |      |

I would like you to do named entity recognition task given above specific entity definitions. They are concepts from materials science literature. Now would you like to recognize entities in an article talking about materials science? Please only provide results in clean JSON format without any irrelevant content. 
Provide your output as below format:
{
"entities": [
    {
    "entity": "entity1 name",
    "category": "entity2 category",
    "context": "entity2 context"
    },
    {
    "entity": "entity2 name",
    "category": "entity2 category",
    "context": "entity2 context"
    }
]
}
In your output, please eliminate content in any format other than json!!!
Again, please remove anything that is not in json format in your output!!!!!!
Please make sure your output is in json format and it should starts with { "entities": [ { and ends with ] }
make sure you have , to separate two entity { }
Below comes the article I would like you to process:
        
"""
        # Below api call is not giving consistent, proper results!

        # response = openai.Completion.create(
        #     model="gpt-3.5-turbo-0125",  
        #     prompt=Schema_And_Prompt + article,
        #     max_tokens=self.max_tokens,  # Adjust max tokens according to your needs
        #     temperature=self.temperature,  # Adjust temperature according to your needs
        #     n=1,  # Number of completions to generate
        #     stop=None,  # Custom stop sequence to end the completion
        # )


        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": Schema_And_Prompt + article}]
            )

        # Get the generated text from the response
        result = response.choices[0].message['content']

        length = len(result)
        print("Result Length:",length)


        if result == "" or result == None or length<100:

            return """
                {
                "entities": [
                {
                "entity": "N/A",
                "category": "N/A",
                "context": "N/A"
                }
                ]
                }
            """


        #result = result.replace("-"," ")
        result = result.replace("\n", " ")
        #non_json_preceding_string = result.split("{")[0]
        #result = result.replace(non_json_preceding_string,"")
        #result = result.replace(", ]}","]}")
        # print('result\n',result)
        return result

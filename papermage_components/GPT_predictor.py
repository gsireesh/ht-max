import openai
import json
import pandas as pd
from dataclasses import dataclass
import difflib
import os
import re
import subprocess
from typing import List, Tuple

from papermage.magelib import (
    Document,
    Entity,
    Metadata,
    Prediction,
    SentencesFieldName,
    Span,
    TokensFieldName,
)
from papermage.predictors import BasePredictor
from papermage.utils.annotate import group_by
from papermage_components.utils import merge_overlapping_entities

@dataclass
class GPTEntity:
    id: str
    entity_type: str
    start: int
    end: int
    entity_string: str

    def to_papermage_entity(self):
        span = Span(self.start, self.end)
        meta = Metadata(
            entity_type=self.entity_type, entity_string=self.entity_string, entity_id=self.id
        )
        return Entity(spans=[span], metadata=meta)
    
def get_offset_map(in_text, out_text, start, end):
    matcher = difflib.SequenceMatcher(isjunk=lambda x: False, a=out_text, b=in_text, autojunk=False)
    opcodes = matcher.get_opcodes()
    offsets = {}

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                offsets[i] = j
        elif tag == "delete":
            assert j1 == j2
            for i in range(i1, i2):
                offsets[i] = j1
        elif tag == "insert":
            # we shouldn't need to do anything here, as long as we only care about matching out to in.
            pass
        elif tag == "replace":

            for i, j in zip(range(i1, i2), range(j1, j2)):
                offsets[i] = None

    # offsets = {v: k for k, v in offsets.items()}
    return offsets


def fix_entity_offsets(entities, offset_map, para_offset, start, end):
    updated_entities = []
    for entity in entities:
        start_offset_file = offset_map[entity.start]
        # try:
        #     end_offset_file = offset_map[entity.end]
        # except KeyError as e:
        #     print(start, end)
        #     raise e

        updated_entities.append(
            GPTEntity(
                entity.id,
                entity.entity_type,
                start_offset_file + para_offset,
                start_offset_file + len(entity.entity_string) + para_offset,
                entity.entity_string,
            )
        )
    return updated_entities


def parse_GPT_content(GPT_content):
    '''
    GPT_content:
    {
    "entities": [
        {
        "entity": "entity name",
        "category": "CategoryName",
        "start": startIndex,
        "end": endIndex
        }
        // Add more entities here
    ]
    }
    '''
    entities = []

    for i,e in enumerate(GPT_content['entities']):
        e_id = i
        e_type, e_start, e_end = e['category'],e['start'],e['end']
        e_string = e['entity']
        entity = GPTEntity(e_id, e_type, int(e_start), int(e_end), e_string)
        entities.append(entity)
    return {"entities": entities}

class GPT_predictor:
    def __init__(self, api_key = "",
                 temperature = 0.7,
                 max_tokens=1000,
                 output_folder) -> None:
        # Set your OpenAI API key
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_folder = output_folder

        # Initialize the NER class instance with the output directory
        print(
            "GPT - NER", "output_folder ", output_folder
        )
        self.curr_file = ""

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [SentencesFieldName, TokensFieldName]
    
    def _predict(self, doc: Document) -> Tuple[Prediction, ...]:

        print("Creating temporary input files")
        input_paragraphs = {}
        input_paragraph_starts = {}
        annotated_sentences = {}
        entities = {}
        for paragraph in doc.reading_order_sections:
            section_name = paragraph.metadata["section_name"]
            paragraph_order = paragraph.metadata["paragraph_reading_order"]
            # TODO: make this more robust!! This implicitly assumes a paragraph has only one span.
            paragraph_text = paragraph.text.replace("\n", " ")
            input_paragraph_starts[(section_name, paragraph_order)] = paragraph.spans[0].start
            # We may want to use the same txt file as matIE
            input_paragraphs[(section_name, paragraph_order)] = paragraph_text
            entities[(section_name, paragraph_order)] =  self.Run_GPT_To_Recognize_Entity(paragraph_text)
            annotated_sentences[(section_name, paragraph_orderd)] = paragraph_text
        # print("Annotating temp files")
        # self.Run_GPT_To_Recognize_Entity() ## need to fix

        # print("Reconciling input and annotated files...")
        # annotated_sentences = {}
        # entities = {}
        # for start, end in input_paragraphs:
        #     folder_name = f'{self.output_folder}{self.curr_file.replace(" ", "_")}'
        #     with open(os.path.join(folder_name, f"{start}-{end}.txt")) as f:
        #         annotated_sentences[(start, end)] = f.read()
        #     entities[(start, end)] = parse_GPT_content(["entities"]

        fixed_entities = []
        for (start, end), input_text in input_paragraphs.items():
            para_offset = input_paragraph_starts[(start, end)]
            annotated_text = annotated_sentences[(start, end)]
            offset_map = get_offset_map(input_text, annotated_text, start, end)
            fixed_entities.extend(
                fix_entity_offsets(entities[(start, end)], offset_map, para_offset, start, end)
            )
        papermage_entities = [entity.to_papermage_entity() for entity in fixed_entities]
        predictions = group_by(
            papermage_entities,
            metadata_field="entity_type",
        )

        return predictions
    
        def Run_GPT_To_Recognize_Entity(self, article):

            
            openai.api_key = self.api_key

            Schema_And_Prompt = """
            I am working on identifying various entities related to materials science within texts. Below are the categories of entities I'm interested in, along with their definitions and examples. Please read the input text and identify entities according to these categories:
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
            For each identified entity, please also provide the category name mentioned above and the span (specific text snippet) that corresponds to it. 
            An example like below:
            {
            "entities": [
                {
                "entity": "entity name",
                "category": "CategoryName",
                "start": startIndex,
                "end": endIndex
                }
                // Add more entities here
            ]
            }


            I would like you to do named entity recognition task given the above specific entity definition. Now would you like to recognize entities in the following text talking about materials science? Please only provide results in clean JSON format without any irrelevant content. 
            """
            # Parameters for the completion
            response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",  # free version of GPT model
            prompt=Schema_And_Prompt + article,
            max_tokens=self.max_tokens,  # Adjust max tokens according to your needs
            temperature=self.temperature,  # Adjust temperature according to your needs
            n=1,  # Number of completions to generate
            stop=None  # Custom stop sequence to end the completion
            )

            # Get the generated text from the response
            result = response.choices[0].text.strip()
            #print('result\n',result)
            output = json.loads(result)
            return output

            
        def generate_df(self, result):
            '''
            result: the output is json object from Run_GPT_To_Recognize_Entity()
            return pandas dataframe 
            '''
            df = pd.DataFrame(result['entities'])
            return df


    # def run_matIE(self):
    #     for dir_name in os.listdir(self.output_folder):
    #         # Rename the folder
    #         if os.path.isdir(os.path.join(self.output_folder, dir_name)) and "copy" not in dir_name:
    #             dir_path = os.path.join(self.output_folder, dir_name)
    #             input_folder = dir_path
    #             output_folder = dir_path

    #             self.process_files_multiprocess(
    #                 self.NER_model_dir,
    #                 self.vocab_dir,
    #                 input_folder,
    #                 output_folder,
    #                 self.gpu_id,
    #                 self.decode_script,
    #             )

    # def process_files_multiprocess(
    #     self, model_dir, vocab_dir, input_folder, output_folder, gpu_id, decode_script
    # ):

    #     env_vars = os.environ.copy()
    #     env_vars["MODEL_DIR"] = model_dir
    #     env_vars["VOCAB_DIR"] = vocab_dir
    #     env_vars["CUDA_VISIBLE_DEVICES"] = ""  # needs to fix later
    #     env_vars["INPUT_DIR"] = os.path.join("../ht-max", input_folder.replace("//", "/"))
    #     env_vars["OUTPUT_DIR"] = os.path.join("../ht-max", output_folder.replace("//", "/"))
    #     env_vars["EXTRA_ARGS"] = ""

    #     subprocess.run(["chmod", "+x", decode_script], check=True)
    #     subprocess.run(
    #         decode_script,
    #         shell=True,
    #         env=env_vars,
    #         cwd="/Users/yuehengzhang/Desktop/Desktop/CMU/NLP/MatIE/",
    #         check=True,
    #     )
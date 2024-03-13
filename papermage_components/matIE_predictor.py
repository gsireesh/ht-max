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
class MatIEEntity:
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
            MatIEEntity(
                entity.id,
                entity.entity_type,
                start_offset_file + para_offset,
                start_offset_file + len(entity.entity_string) + para_offset,
                entity.entity_string,
            )
        )
    return updated_entities


def parse_ann_content(ann_content):
    entities = []
    for line in ann_content.split("\n"):
        if line.startswith("T"):
            parts = line.split("\t")
            e_id = parts[0]
            e_type, e_start, e_end = parts[1].split()
            e_string = "\t".join(parts[2:])
            entity = MatIEEntity(e_id, e_type, int(e_start), int(e_end), e_string)
            entities.append(entity)
    return {"entities": entities}


class MatIEPredictor(BasePredictor):
    def __init__(
        self,
        NER_model_dir="",
        vocab_dir="",
        output_folder="",
        gpu_id="0",
        decode_script="",
    ):
        self.NER_model_dir = NER_model_dir
        self.vocab_dir = vocab_dir
        self.output_folder = output_folder
        self.gpu_id = gpu_id
        self.decode_script = decode_script

        # Initialize the NER class instance with the output directory
        print(
            "NER_model_dir ", NER_model_dir, "vocab_dir", vocab_dir, "output_folder ", output_folder
        )
        self.curr_file = ""

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [SentencesFieldName, TokensFieldName]

    def _predict(self, doc: Document) -> Tuple[Prediction, ...]:

        print("Creating temporary input files")
        input_paragraphs = {}
        input_paragraph_starts = {}
        for paragraph in doc.reading_order_sections:
            section_name = paragraph.metadata["section_name"]
            paragraph_order = paragraph.metadata["paragraph_reading_order"]
            # TODO: make this more robust!! This implicitly assumes a paragraph has only one span.
            paragraph_text = paragraph.text.replace("\n", " ")
            input_paragraph_starts[(section_name, paragraph_order)] = paragraph.spans[0].start

            input_paragraphs[(section_name, paragraph_order)] = self.generate_txt(
                self.curr_file, paragraph_text, section_name, paragraph_order
            )

        print("Annotating temp files")
        self.run_matIE()

        print("Reconciling input and annotated files...")
        annotated_sentences = {}
        entities = {}
        for start, end in input_paragraphs:
            folder_name = f'{self.output_folder}{self.curr_file.replace(" ", "_")}'
            with open(os.path.join(folder_name, f"{start}-{end}.txt")) as f:
                annotated_sentences[(start, end)] = f.read()
            with open(os.path.join(folder_name, f"{start}-{end}.ann")) as f:
                entities[(start, end)] = parse_ann_content(f.read())["entities"]

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

    def generate_txt(self, filename, paragraph_text, section_name, reading_order):
        folder_name = f'{self.output_folder}{filename.replace(" ", "_")}'
        file_path = f"{folder_name}/{section_name}-{reading_order}.txt"
        os.makedirs(folder_name, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(paragraph_text)

        os.makedirs(folder_name + "_copy", exist_ok=True)
        with open(
            file_path.replace(folder_name, folder_name + "_copy"), "w", encoding="utf-8"
        ) as file:
            file.write(paragraph_text)

        return paragraph_text

    def run_matIE(self):
        for dir_name in os.listdir(self.output_folder):
            # Rename the folder
            if os.path.isdir(os.path.join(self.output_folder, dir_name)) and "copy" not in dir_name:
                dir_path = os.path.join(self.output_folder, dir_name)
                input_folder = dir_path
                output_folder = dir_path

                self.process_files_multiprocess(
                    self.NER_model_dir,
                    self.vocab_dir,
                    input_folder,
                    output_folder,
                    self.gpu_id,
                    self.decode_script,
                )

    def process_files_multiprocess(
        self, model_dir, vocab_dir, input_folder, output_folder, gpu_id, decode_script
    ):

        env_vars = os.environ.copy()
        env_vars["MODEL_DIR"] = model_dir
        env_vars["VOCAB_DIR"] = vocab_dir
        env_vars["CUDA_VISIBLE_DEVICES"] = ""  # needs to fix later
        env_vars["INPUT_DIR"] = os.path.join("../ht-max", input_folder.replace("//", "/"))
        env_vars["OUTPUT_DIR"] = os.path.join("../ht-max", output_folder.replace("//", "/"))
        env_vars["EXTRA_ARGS"] = ""

        subprocess.run(["chmod", "+x", decode_script], check=True)
        subprocess.run(
            decode_script,
            shell=True,
            env=env_vars,
            cwd="/Users/sireeshgururaja/src/MatIE",
            check=True,
        )

    # Example usage
    # matie = MatIE(NER_model_dir="path/to/model", vocab_dir="path/to/vocab", output_folder="path/to/output", gpu_id="0", decode_script="path/to/decode.sh")
    # matie.run_matIE()
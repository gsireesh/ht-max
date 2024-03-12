from dataclasses import dataclass
import difflib
import os
import re
import subprocess
from typing import List

from papermage.magelib import (
    Document,
    Entity,
    Metadata,
    SentencesFieldName,
    Span,
    TokensFieldName,
)
from papermage.predictors import BasePredictor


@dataclass
class MatIEEntity:
    id: str
    entity_type: str
    start: int
    end: int
    entity_string: str

    def to_papermage_entity(self):
        span = Span(self.start, self.end)
        meta = Metadata(entity_type=self.entity_type, entity_string=self.entity_string)
        return Entity(
            spans=[span],
        )


def get_offset_map(in_text, out_text):
    matcher = difflib.SequenceMatcher(isjunk=lambda x: False, a=out_text, b=in_text, autojunk=False)
    opcodes = matcher.get_opcodes()

    current_offset = 0
    offsets = {}

    for (tag, i1, i2, j1, j2) in opcodes:
        if tag == "equal":
            assert i2 - i1 == j2 - j1
            for i, j in zip(range(i1, i2), range(j1, j2)):
                offsets[i] = j + current_offset
        if tag == "delete":
            for i in range(i1, i2):
                offsets[i] = None
        if tag == "replace":
            if not i2 - i1 == j2 - j1:
                raise AssertionError("replacement sections not the same length")
            for i, j in zip(range(i1, i2), range(j1, j2)):
                offsets[i] = None
        # do not need to worry about insertions.

    # offsets = {v: k for k, v in offsets.items()}
    return offsets


def fix_entity_offsets(entities, offset_map, para_offset):
    updated_entities = []
    for entity in entities:
        start_offset_file = offset_map[entity.start]
        end_offset_file = offset_map[entity.end]

        updated_entities.append(
            MatIEEntity(
                entity.id,
                entity.entity_type,
                start_offset_file + para_offset,
                end_offset_file + para_offset,
                entity.entity_string,
            )
        )
    return updated_entities


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

    def _predict(self, doc: Document) -> List[Entity]:

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
        for (start, end) in input_paragraphs:
            folder_name = f'{self.output_folder}{self.curr_file.replace(" ", "_")}'
            with open(os.path.join(folder_name, f"{start}-{end}.txt")) as f:
                annotated_sentences[(start, end)] = f.read()
            with open(os.path.join(folder_name, f"{start}-{end}.ann")) as f:
                entities[(start, end)] = self.parse_ann_content(f.read())["entities"]

        fixed_entities = []
        for (start, end), input_text in input_paragraphs.items():
            para_offset = input_paragraph_starts[(start, end)]
            annotated_text = annotated_sentences[(start, end)]
            offset_map = get_offset_map(input_text, annotated_text)
            fixed_entities.extend(
                fix_entity_offsets(entities[(start, end)], offset_map, para_offset)
            )
        return [entity.to_papermage_entity() for entity in fixed_entities]

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

        return "\n".join(paragraph_text)

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

    def parse_ann_content(self, ann_content):
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

    # Example usage
    # matie = MatIE(NER_model_dir="path/to/model", vocab_dir="path/to/vocab", output_folder="path/to/output", gpu_id="0", decode_script="path/to/decode.sh")
    # matie.run_matIE()

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

    def parse_ann_files(self, directory):
        span_list = []

        # Regex to extract start value from file name
        file_name_regex = re.compile(r"(\d+)-\d+\.ann")

        # Iterate over all .ann files in the specified directory
        for file_name in os.listdir(directory):
            if file_name.endswith(".ann"):
                match = file_name_regex.search(file_name)
                if match:
                    start_offset = int(match.group(1))

                    file_path = os.path.join(directory, file_name)
                    with open(file_path, "r") as file:
                        for line in file:
                            if line.startswith("T"):
                                parts = line.split("\t")
                                if len(parts) >= 3:
                                    entity_info = parts[1].split()
                                    if len(entity_info) >= 3:
                                        # Adjust span positions with start_offset
                                        adjusted_start = int(entity_info[1]) + start_offset
                                        adjusted_end = int(entity_info[2]) + start_offset
                                        # Add the adjusted span to the list
                                        span_list.append([adjusted_start, adjusted_end])

        return span_list

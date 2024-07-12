from dataclasses import dataclass
import difflib
import os
import re
import subprocess
from tempfile import TemporaryDirectory
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

NER_MODEL_RELATIVE_PATH = "model"
VOCAB_RELATIVE_PATH = "vpack_mat"
DECODE_SCRIPT_RELATIVE_PATH = "decode.sh"
WORKING_DIRECTORY = "data/matIE_annotation"


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


relation_re = re.compile("R\d+\t(?P<r_type>.*) Arg1:(?P<arg1>T\d+) Arg2:(?P<arg2>T\d+)")


def get_offset_map(in_text, out_text):
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


def fix_entity_offsets(entities, offset_map, para_offset):
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
    relations = []
    for line in ann_content.split("\n"):
        if line.startswith("T"):
            parts = line.split("\t")
            e_id = parts[0]
            e_type, e_start, e_end = parts[1].split()
            e_string = "\t".join(parts[2:])
            entity = MatIEEntity(e_id, e_type, int(e_start), int(e_end), e_string)
            entities.append(entity)
        elif line.startswith("R"):
            match = relation_re.fullmatch(line)
            relations.append(
                {
                    "relation_type": match.group("r_type"),
                    "arg1": match.group("arg1"),
                    "arg2": match.group("arg2"),
                }
            )

    return {"entities": entities, "relations": relations}


class MatIEPredictor(BasePredictor):
    def __init__(
        self,
        matIE_directory,
        gpu_id="0",
    ):
        self.matIE_directory = matIE_directory
        self.NER_model_dir = os.path.join(matIE_directory, NER_MODEL_RELATIVE_PATH)
        self.vocab_dir = os.path.join(matIE_directory, VOCAB_RELATIVE_PATH)
        self.decode_script = os.path.join(matIE_directory, DECODE_SCRIPT_RELATIVE_PATH)

        self.working_folder = WORKING_DIRECTORY
        if not os.path.exists(self.working_folder):
            os.makedirs(self.working_folder, exist_ok=True)
        self.gpu_id = gpu_id
        self.preferred_layer_name = "TAGGED_ENTITIES_MatIE"

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [SentencesFieldName, TokensFieldName]

    def _predict(self, doc: Document) -> List[Entity]:

        print("Creating temporary input files")

        doc_temp_folder = TemporaryDirectory(
            dir=self.working_folder, prefix="matie_file_annotation_"
        )

        input_paragraphs = {}
        input_paragraph_starts = {}
        for paragraph in doc.reading_order_sections:
            section_name = paragraph.metadata["section_name"]
            paragraph_order = paragraph.metadata["paragraph_reading_order"]
            # TODO: make this more robust!! This implicitly assumes a paragraph has only one span.
            paragraph_text = paragraph.text.replace("\n", " ")
            if len(paragraph.spans) != 0:
                input_paragraph_starts[(section_name, paragraph_order)] = paragraph.spans[0].start

                input_paragraphs[(section_name, paragraph_order)] = self.generate_txt(
                    doc_temp_folder.name, paragraph_text, section_name, paragraph_order
                )

        print("Annotating temp files")
        self.run_matIE()

        print("Reconciling input and annotated files...")
        annotated_sentences = {}
        entities = {}
        relations = {}
        for section_name, paragraph in input_paragraphs:
            folder_name = doc_temp_folder.name
            with open(
                os.path.join(folder_name, f"{section_name}-{paragraph}.txt".replace("/", "_"))
            ) as f:
                annotated_sentences[(section_name, paragraph)] = f.read()
            with open(
                os.path.join(folder_name, f"{section_name}-{paragraph}.ann".replace("/", "_"))
            ) as f:
                ann_content = parse_ann_content(f.read())
                entities[(section_name, paragraph)] = ann_content["entities"]
                relations[(section_name, paragraph)] = ann_content["relations"]

        fixed_entities = []
        for (key, input_text), paragraph in zip(
            input_paragraphs.items(), doc.reading_order_sections
        ):
            para_offset = input_paragraph_starts[key]
            annotated_text = annotated_sentences[key]
            offset_map = get_offset_map(input_text, annotated_text)
            fixed_entities.extend(fix_entity_offsets(entities[key], offset_map, para_offset))
            paragraph.metadata["in_section_relations"] = relations[key]
        papermage_entities = [entity.to_papermage_entity() for entity in fixed_entities]
        return papermage_entities

    def generate_txt(self, folder_name, paragraph_text, section_name, reading_order):
        file_path = os.path.join(
            folder_name, f"{section_name}-{reading_order}.txt".replace("/", "_")
        )
        os.makedirs(folder_name, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(paragraph_text)

        return paragraph_text

    def run_matIE(self):
        for dir_name in os.listdir(self.working_folder):
            if (
                not os.path.isdir(os.path.join(self.working_folder, dir_name))
                or "_original" in dir_name
            ):
                continue

            dir_path = os.path.join(self.working_folder, dir_name)
            input_folder = dir_path
            output_folder = dir_path

            self.process_files_multiprocess(
                input_folder,
                output_folder,
            )

    def process_files_multiprocess(self, input_folder, output_folder):

        env_vars = os.environ.copy()
        env_vars["MODEL_DIR"] = self.NER_model_dir
        env_vars["VOCAB_DIR"] = self.vocab_dir
        env_vars["CUDA_VISIBLE_DEVICES"] = ""  # needs to fix later
        # bizarrely, taking out the `../ht-max` from these paths breaks something in MatIE
        env_vars["INPUT_DIR"] = os.path.join("../ht-max", input_folder.replace("//", "/"))
        env_vars["OUTPUT_DIR"] = os.path.join("../ht-max", output_folder.replace("//", "/"))
        env_vars["EXTRA_ARGS"] = ""

        subprocess.run(["chmod", "+x", self.decode_script], check=True)
        subprocess.run(
            self.decode_script,
            shell=True,
            env=env_vars,
            cwd=self.matIE_directory,
            check=True,
        )

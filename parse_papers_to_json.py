import json
import logging
import os
import sys

import fire
from tqdm.auto import tqdm

from papermage import Document
from papermage_components.materials_recipe import MaterialsRecipe


def get_doc_title(document: Document):
    """In observation, sometimes PaperMage picks up on fragments of the journal title.
    This function takes the longest title, which tends to be the real one."""
    document_title = ""
    for title in document.titles:
        if len(title.text) > len(document_title):
            document_title = title.text
    return document_title


def parse_papers_to_json(input_folder: str, output_folder: str, overwrite_if_present: bool = False):
    recipe = MaterialsRecipe(
        NER_model_dir="/Users/sireeshgururaja/src/MatIE/model",  # the directory of the NER model
        vocab_dir="/Users/sireeshgururaja/src/MatIE/vpack_mat",  # the directory of the vocabulary
        output_folder="data/AM_Creep_Papers_parsed_test/",  # the directory of the vocabulary
        gpu_id="0",
        decode_script="/Users/sireeshgururaja/src/MatIE/decode.sh",  # the decode module
    )

    pdf_list = [
        pdf_filename
        for pdf_filename in os.listdir(input_folder)
        if pdf_filename.lower().endswith(".pdf")
    ]

    failed_files = []
    for pdf_filename in tqdm(pdf_list):
        output_path = os.path.join(output_folder, pdf_filename.lower().replace(".pdf", ".json"))

        if os.path.exists(output_path) and not overwrite_if_present:
            print(f"File {output_path} already exists! Skipping parsing.")
            continue

        try:
            parsed_paper = recipe.from_pdf(os.path.join(input_folder, pdf_filename))
            with open(output_path, "w") as f:
                json.dump(parsed_paper.to_json(), f, indent=4)
        except Exception as e:
            logging.error(f"Failed to parse paper {pdf_filename}", exc_info=True)
            failed_files.append({"filename": pdf_filename, "error_class": str(e)})

    with open("data/failed_files.json", "w") as f:
        json.dump(failed_files, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(parse_papers_to_json)

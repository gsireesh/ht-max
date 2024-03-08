import json
import logging
import os

import fire

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


def parse_papers_to_json(input_folder: str, output_folder: str):
    recipe = MaterialsRecipe()

    pdf_list = [
        pdf_filename
        for pdf_filename in os.listdir(input_folder)
        if pdf_filename.lower().endswith(".pdf")
    ]

    paper_list = [
        recipe.from_pdf(os.path.join(input_folder, pdf_filename)) for pdf_filename in pdf_list
    ]
    for paper, pdf_filename in zip(paper_list, pdf_list):
        with open(
            os.path.join(output_folder, pdf_filename.lower().replace(".pdf", ".json")), "w"
        ) as f:
            json.dump(
                paper.to_json(),
                f,
                indent=4,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(parse_papers_to_json)

import itertools
import json
import logging
import math
import os

import fire
from matplotlib import pyplot as plt
import pandas as pd
import papermage
from papermage import Document
from papermage.recipes import CoreRecipe
from papermage.visualizers import plot_entities_on_page
from tqdm.auto import tqdm


def get_doc_title(document: Document):
    """In observation, sometimes PaperMage picks up on fragments of the journal title.
    This function takes the longest title, which tends to be the real one."""
    document_title = ""
    for title in document.titles:
        if len(title.text) > len(document_title):
            document_title = title.text
    return document_title


def parse_papers_to_json(input_folder: str, output_folder: str):
    recipe = CoreRecipe()

    paper_list = [
        recipe.from_pdf(os.path.join(input_folder, pdf_filename))
        for pdf_filename in os.listdir(input_folder)
        if pdf_filename.lower().endswith(".pdf")
    ]
    for paper in paper_list:
        paper_title = get_doc_title(paper)
        paper_sentences = [sentence.text.replace("\n", " ") for sentence in paper.sentences]
        paper_abstract = paper.abstracts[0].text
        section_text = {}

        with open(
            os.path.join(output_folder, paper_title.replace(" ", "_") + ".json"), "w"
        ) as f:
            json.dump(
                {
                    "title": paper_title,
                    "abstract": paper_abstract,
                    "sentences": paper_sentences,
                    "section_text": section_text,
                },
                f,
                indent=4,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(parse_papers_to_json)

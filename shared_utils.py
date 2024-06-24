import json
import os

from papermage import Document
from papermage.visualizers import plot_entities_on_page
import streamlit as st
import spacy

PARSED_PAPER_FOLDER = "data/Midyear_Review_Papers_Parsed"


@st.cache_resource
def load_document(doc_filename):
    with open(os.path.join(PARSED_PAPER_FOLDER, doc_filename)) as f:
        document = Document.from_json(json.load(f))
    return document


def highlight_section_on_page(document, page_number, section_name, paragraph):
    page = document.pages[page_number]
    page_image = page.images[0]
    section_entities = [
        e
        for e in page.reading_order_sections
        if e.metadata["section_name"] == section_name
        and e.metadata["paragraph_reading_order"] == paragraph
    ]
    highlighted = plot_entities_on_page(
        page_image,
        section_entities,
        box_width=2,
        box_alpha=0.2,
        box_color="green",
        page_number=page_number,
    )
    return highlighted


@st.cache_resource
def get_spacy_pipeline():
    return spacy.load("en_core_sci_md", exclude=["tagger", "parser", "ner", "lemmatizer"])

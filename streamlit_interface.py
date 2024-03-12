import json
import os

from papermage import Document
from papermage.visualizers import plot_entities_on_page
import streamlit as st

st.set_page_config(layout="wide")

# CONSTANTS
PARSED_PAPER_FOLDER = "data/AM_Creep_Papers_parsed"

# focus_document = None


## HELPER FUNCTIONS
def load_document(doc_filename):
    with open(os.path.join(PARSED_PAPER_FOLDER, doc_filename)) as f:
        document = Document.from_json(json.load(f))
    return document


def highlight_section_on_page(document, page_number, section_name):
    page = document.pages[page_number]
    page_image = page.images[0]
    section_entities = [
        e for e in page.reading_order_sections if e.metadata["section_name"] == section_name
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


file_options = os.listdir(PARSED_PAPER_FOLDER)

with st.sidebar:  # .form("File selector"):
    st.write("Select a pre-parsed file whose results to display")
    file_selector = st.selectbox("Parsed file", options=file_options)
    focus_document = load_document(file_selector)

if focus_document is not None:
    doc_vis_column, sections_column = st.columns([0.6, 0.4])

    with doc_vis_column:
        focus_page = st.slider(
            label="Select the document page to view",
            min_value=1,
            max_value=len(focus_document.pages),
            value=1,
            format="Page %d",
        )
        focus_page = focus_page - 1

    with sections_column:
        section_titles = [
            section.metadata["section_name"]
            for section in focus_document.pages[focus_page].reading_order_sections
        ]
        selected_section = st.selectbox(
            label="Select a section displayed on this page:", options=section_titles
        )
        for entity in focus_document.pages[focus_page].reading_order_sections:
            if entity.metadata["section_name"] == selected_section:
                st.text_area(entity.text)

    doc_vis_column.write(focus_page)
    highlighted_image = highlight_section_on_page(focus_document, focus_page, selected_section)
    doc_vis_column.write(highlighted_image.pilimage)

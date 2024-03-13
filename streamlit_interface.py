import json
import os

from papermage import Document
from papermage.visualizers import plot_entities_on_page
import spacy
import spacy_streamlit
import streamlit as st

from papermage_components.utils import visualize_paragraph
from papermage_components.constants import MAT_IE_TYPES, MAT_IE_COLORS


st.set_page_config(layout="wide")

# CONSTANTS
PARSED_PAPER_FOLDER = "data/AM_Creep_Papers_parsed"

# focus_document = None


## HELPER FUNCTIONS
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
    return spacy.load("en_core_sci_md", exclude=["tagger", "parser", "ner"])


file_options = os.listdir(PARSED_PAPER_FOLDER)

with st.sidebar:  # .form("File selector"):
    st.write("Select a pre-parsed file whose results to display")
    file_selector = st.selectbox("Parsed file", options=file_options)
    focus_document = load_document(file_selector)

if focus_document is not None:
    doc_vis_column, sections_column = st.columns([0.4, 0.6])

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
            (section.metadata["section_name"], section.metadata["paragraph_reading_order"])
            for section in focus_document.pages[focus_page].reading_order_sections
        ]
        selected_section = st.selectbox(
            label="Select a section and paragraph displayed on this page:",
            options=section_titles,
            format_func=lambda x: f"{x[0]}, Paragraph {x[1]}",
        )

        section_name, paragraph = selected_section
        section_entities = [
            e
            for e in focus_document.pages[focus_page].reading_order_sections
            if e.metadata["section_name"] == section_name
            and e.metadata["paragraph_reading_order"] == paragraph
        ]
        st.markdown(f"## {section_entities[0].metadata['section_name']}")
        for entity in section_entities:
            st.write(entity.text)
            st.markdown("---")
            spacy_doc = visualize_paragraph(entity, get_spacy_pipeline())
            spacy_streamlit.visualize_ner(
                spacy_doc,
                labels=MAT_IE_TYPES,
                show_table=True,
                title="With MatIE entities",
                displacy_options={"colors": MAT_IE_COLORS},
            )
            # for annotation_key in MAT_IE_TYPES:
            #     annotations = entity.intersect_by_span(annotation_key)
            #     st.markdown(f"**{annotation_key}:**")
            #     st.write([a.text for a in annotations])

    highlighted_image = highlight_section_on_page(
        focus_document, focus_page, section_name, paragraph
    )
    doc_vis_column.image(highlighted_image.pilimage, use_column_width=True)

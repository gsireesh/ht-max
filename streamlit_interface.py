import json
import os

import pandas as pd
from papermage import Document
from papermage import Box
from papermage.visualizers import plot_entities_on_page
import spacy
import spacy_streamlit
import streamlit as st
from streamlit.column_config import TextColumn
from streamlit_dimensions import st_dimensions
from streamlit_image_coordinates import streamlit_image_coordinates

from papermage_components.utils import visualize_matIE_annotations, visualize_highlights
from papermage_components.constants import MAT_IE_TYPES, MAT_IE_COLORS
from papermage_components.highlightParser import HIGHLIGHT_TYPES


st.set_page_config(layout="wide")

# CONSTANTS
PARSED_PAPER_FOLDER = "data/AM_Creep_Papers_parsed"
BOX_PADDING = 0.01

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


def get_matie_entities(doc, allowed_sections, allowed_types):
    all_entities = []
    for matie_type in MAT_IE_TYPES:
        if matie_type not in allowed_types:
            continue
        entities = getattr(doc, matie_type)
        for entity in entities:
            section_name = entity.reading_order_sections[0].metadata["section_name"]
            if section_name not in allowed_sections:
                continue
            sentence_context = entity.sentences[0].text
            all_entities.append(
                {
                    "entity_type": matie_type,
                    "entity_text": entity.text,
                    "entity_section": section_name,
                    "sentence_context": sentence_context,
                }
            )
    return all_entities


def get_tables(doc, filter_string):
    tables_to_return = []
    for table in doc.tables:
        table_dict = table.metadata["table_dict"]
        filter_match = any([filter_string in header for header in table_dict])
        if not table_dict or not filter_match:
            continue
        tables_to_return.append(table)
    return tables_to_return


@st.cache_resource
def get_spacy_pipeline():
    return spacy.load("en_core_sci_md", exclude=["tagger", "parser", "ner", "lemmatizer"])


file_options = os.listdir(PARSED_PAPER_FOLDER)

with st.sidebar:  # .form("File selector"):
    st.write("Select a pre-parsed file whose results to display")
    file_selector = st.selectbox("Parsed file", options=file_options)
    focus_document = load_document(file_selector)

summary_view, annotations_view, inspection_view = st.tabs(
    ["Summary View", "Annotations View", "Inspection View"]
)

with summary_view:
    entities_column, table_column = st.columns([0.5, 0.5])
    with entities_column:
        st.write("## Tagged Entities")
        all_sections = {e.metadata["section_name"] for e in focus_document.reading_order_sections}
        entity_type_choice = st.multiselect(
            label="Choose which entity types to display", options=MAT_IE_TYPES, default=None
        )
        section_choice = st.multiselect(
            label="Choose sections from which to display entities",
            options=all_sections,
            default=None,
        )
        entities = get_matie_entities(
            focus_document, allowed_sections=section_choice, allowed_types=entity_type_choice
        )
        st.write(f"Found {len(entities)} entities:")
        st.dataframe(
            pd.DataFrame(entities),
            hide_index=True,
            use_container_width=True,
            column_config={
                "sentence_context": TextColumn(label="Sentence Context", width="large"),
                "entity_type": TextColumn("Entity Type", width=None),
                "entity_text": TextColumn("Text", width=None),
                "entity_section": TextColumn("Section", width=None),
            },
        )

    with table_column:
        st.write("## Parsed Tables")
        column_header_filter = st.text_input(
            "table_filter",
            label_visibility="collapsed",
            placeholder="Filter tables based on column headers:",
        )

        filtered_tables = get_tables(focus_document, column_header_filter)
        st.write(f"Found {len(filtered_tables)} matching tables")
        for table in filtered_tables:
            with st.container(border=True):
                table_dict = table.metadata["table_dict"]
                st.dataframe(pd.DataFrame(table_dict))
                table_page = table.boxes[0].page
                st.write(f"From page {table_page + 1}")


with annotations_view:
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

    clicked_section = st.session_state.get("clicked_section", None)
    if (
        clicked_section is None
        or clicked_section[0] != file_selector
        or clicked_section[1] != focus_page
    ):
        st.session_state["clicked_section"] = None
        section_name = None
        paragraph = None
    else:
        section_name = clicked_section[2]
        paragraph = clicked_section[3]

    with sections_column:
        if section_name is not None:
            section_entities = [
                e
                for e in focus_document.pages[focus_page].reading_order_sections
                if e.metadata["section_name"] == section_name
                and e.metadata["paragraph_reading_order"] == paragraph
            ]
            st.markdown(f"## {section_entities[0].metadata['section_name']}")

            for entity in section_entities:
                entity_highlights = entity.annotation_highlights
                highlight_spacy_doc = visualize_highlights(entity, get_spacy_pipeline())
                spacy_streamlit.visualize_ner(
                    highlight_spacy_doc,
                    labels=HIGHLIGHT_TYPES,
                    show_table=False,
                    title="Highlighted Entities",
                    # displacy_options={"colors": MAT_IE_COLORS},
                    key="annotation_highlights",
                )
                st.markdown("---")
                matIE_spacy_doc = visualize_matIE_annotations(entity, get_spacy_pipeline())
                spacy_streamlit.visualize_ner(
                    matIE_spacy_doc,
                    labels=MAT_IE_TYPES,
                    show_table=False,
                    title="MatIE entities",
                    displacy_options={"colors": MAT_IE_COLORS},
                    key="matIE_highlights",
                )

    with doc_vis_column:
        st.write("Click a section of text to view the annotations on it:")
        highlighted_image = highlight_section_on_page(
            focus_document, focus_page, section_name, paragraph
        )
        page_width, page_height = highlighted_image.pilimage.size
        ratio = page_height / page_width

        image_width = st_dimensions(key="doc_vis")["width"]
        image_height = image_width * ratio

        image_coords = streamlit_image_coordinates(
            highlighted_image.pilimage, key="pil", width=image_width
        )
        x = image_coords.get("x", 0) / image_width
        y = image_coords.get("y", 0) / image_height

        click_sections = focus_document.find(
            Box(x - BOX_PADDING / 2, y - BOX_PADDING / 2, BOX_PADDING, BOX_PADDING, focus_page),
            "reading_order_sections",
        )
        if not click_sections:
            st.toast("No parsed content found at specified location!")
        else:
            section_name = click_sections[0].metadata["section_name"]
            paragraph = click_sections[0].metadata["paragraph_reading_order"]
            if st.session_state.get("clicked_section") != (
                file_selector,
                focus_page,
                section_name,
                paragraph,
            ):
                st.session_state["clicked_section"] = (
                    file_selector,
                    focus_page,
                    section_name,
                    paragraph,
                )
                st.rerun()

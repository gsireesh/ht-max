import pandas as pd
import spacy_streamlit
from streamlit_dimensions import st_dimensions
from streamlit_image_coordinates import streamlit_image_coordinates

from papermage import Box, Entity
from papermage.visualizers import plot_entities_on_page

from papermage_components.constants import (
    HIGHLIGHT_COLORS,
    HIGHLIGHT_TYPES,
    MAT_IE_COLORS,
    MAT_IE_TYPES,
)
from papermage_components.utils import (
    visualize_highlights,
    get_table_images,
)
from interface_utils import *

st.set_page_config(layout="wide")

# CONSTANTS
BOX_PADDING = 0.01

file_options = os.listdir(PARSED_PAPER_FOLDER)


LAYER_EXCLUDES = ["symbols", "images", "metadata"]

focus_entities = None


def get_layers_with_boxes(document) -> list[str]:
    return [
        layer
        for layer in document.layers
        if layer not in LAYER_EXCLUDES and all([e.boxes for e in getattr(document, layer)])
    ]


with st.sidebar:  # .form("File selector"):
    st.write("Select a parsed file whose results to display")
    focus_file = st.session_state.get("focus_document")
    file_selector = st.selectbox(
        "Parsed file",
        options=file_options,
        index=file_options.index(focus_file) if focus_file else 0,
    )
    st.session_state["focus_document"] = file_selector
    focus_document = load_document(file_selector)

    st.divider()
    st.write("Select layer from PaperMage pipeline to visualize:")
    focus_layer = st.selectbox("Layer", options=get_layers_with_boxes(focus_document))


doc_vis_column, inspection_column = st.columns([0.4, 0.6])
with doc_vis_column:
    focus_page = st.slider(
        label="Select the document page to view",
        min_value=1,
        max_value=len(focus_document.pages),
        value=1,
        format="Page %d",
    )
    focus_page = focus_page - 1

    st.write("Click a section of text to view the annotations on it:")

    if (coords := st.session_state.get("clicked_coordinates")) and coords[2] == focus_page:
        x, y, page = coords
        focus_entities = focus_document.find(
            Box(x - BOX_PADDING / 2, y - BOX_PADDING / 2, BOX_PADDING, BOX_PADDING, focus_page),
            focus_layer,
        )
        highlighted_image = highlight_entities_on_page(
            focus_document, focus_page, focus_entities, selectable_layers=[focus_layer]
        )
    else:
        highlighted_image = plot_selectable_regions(
            focus_document, focus_page, selectable_layers=[focus_layer]
        )

    page_width, page_height = highlighted_image.pilimage.size
    ratio = page_height / page_width

    image_width = st_dimensions(key="doc_vis")["width"]
    image_height = image_width * ratio

    image_coords = streamlit_image_coordinates(
        highlighted_image.pilimage, key="annotation_page_image", width=image_width
    )

    if image_coords is not None:
        x = image_coords["x"] / image_width
        y = image_coords["y"] / image_height

        st.session_state["clicked_coordinates"] = (x, y, focus_page)

with inspection_column:
    if not focus_entities:
        st.write("Select an entity on the PDF to visualize its contents.")
    else:
        with st.container(border=True):
            st.write("### Entity Images")
            for entity in focus_entities:
                entity_images = get_table_images(entity, focus_document)
                for entity_image in entity_images:
                    st.image(entity_image)
        with st.container(border=True):
            st.write("### Entity Text")
            show_sentences = st.toggle("Split into sentences")
            for entity in focus_entities:
                if show_sentences:
                    for sentence in entity.sentences:
                        st.write(sentence.text)
                else:
                    st.write(focus_entities[0].text)
                st.divider()

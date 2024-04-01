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
    visualize_matIE_annotations,
    get_table_image,
)
from shared_utils import *

st.set_page_config(layout="wide")

# CONSTANTS
BOX_PADDING = 0.01

file_options = os.listdir(PARSED_PAPER_FOLDER)


def visualize_table_with_boxes(table, doc, include_tokens):
    table_box = table.boxes[0]
    table_boxes = [Box.from_json(b) for b in table.metadata["cell_boxes"]]
    vis_entity = plot_entities_on_page(
        doc.pages[table_box.page].images[0],
        entities=[Entity(boxes=table_boxes)],
        box_width=2,
        box_color="cornflowerblue",
    )
    if include_tokens:
        vis_entity = plot_entities_on_page(
            vis_entity, entities=table.tokens, box_width=2, box_color="red"
        )
    vis_entity = get_table_image(table, doc, vis_entity.pilimage)
    return vis_entity


with st.sidebar:  # .form("File selector"):
    st.write("Select a pre-parsed file whose results to display")
    file_selector = st.selectbox("Parsed file", options=file_options)
    focus_document = load_document(file_selector)

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
    if isinstance(section_name, str):
        section_entities = [
            e
            for e in focus_document.pages[focus_page].reading_order_sections
            if e.metadata["section_name"] == section_name
            and e.metadata["paragraph_reading_order"] == paragraph
        ]
        st.markdown(
            f"## Section: {section_entities[0].metadata['section_name']}, paragraph "
            f"{section_entities[0].metadata['paragraph_reading_order']}"
        )

        for entity in section_entities:
            entity_highlights = getattr(entity, "annotation_highlights", [])
            highlight_spacy_doc = visualize_highlights(entity, get_spacy_pipeline())
            st.write("### Highlighted Entities")
            spacy_streamlit.visualize_ner(
                highlight_spacy_doc,
                labels=HIGHLIGHT_TYPES,
                show_table=False,
                displacy_options={"colors": HIGHLIGHT_COLORS},
                key="annotation_highlights",
                title=None,
            )
            st.markdown("---")
            st.write("### MatIE Entities")
            show_matie_entities = st.toggle("Show MatIE Entities", value=True)
            if show_matie_entities:
                matIE_spacy_doc = visualize_matIE_annotations(entity, get_spacy_pipeline())
                spacy_streamlit.visualize_ner(
                    matIE_spacy_doc,
                    labels=MAT_IE_TYPES,
                    show_table=False,
                    displacy_options={"colors": MAT_IE_COLORS},
                    key="matIE_highlights",
                    title=None,
                )
            st.markdown("---")
            st.write("### GPT-3.5 Entities")
            show_gpt_entities = st.toggle("Show GPT-3.5 Entities", value=True)
            if show_gpt_entities:
                gpt_entities = entity.metadata.get("gpt_recognized_entities")
                if gpt_entities:
                    formatted_entities = []
                    for e in gpt_entities:
                        entity_type = e.get("entity_type").replace(" ", "_")

                        formatted_entities.append(
                            {
                                "entity_type": entity_type,
                                "entity_text": e.get("entity_string"),
                                "sentence_context": e.get("entity_context"),
                            }
                        )
                    st.dataframe(pd.DataFrame(gpt_entities))

    # table by id
    elif isinstance(section_name, int):
        table = focus_document.tables[section_name]
        st.write("### Table with model annotations:")
        show_tokens = st.checkbox("Show Tokens")
        table_visualized = visualize_table_with_boxes(table, focus_document, show_tokens)
        st.write(table_visualized)
        st.write("### Parsed Table:")
        st.dataframe(pd.DataFrame(table.metadata["table_dict"]), hide_index=True)


with doc_vis_column:
    st.write("Click a section of text to view the annotations on it:")
    if isinstance(section_name, str):
        highlighted_image = highlight_section_on_page(
            focus_document, focus_page, section_name, paragraph
        )
    elif isinstance(section_name, int):
        highlighted_image = plot_entities_on_page(
            focus_document.pages[focus_page].images[0],
            entities=[focus_document.tables[section_name]],
            box_width=2,
            box_alpha=0.2,
            box_color="green",
        )
    else:
        highlighted_image = focus_document.pages[focus_page].images[0]
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

        click_sections = focus_document.find(
            Box(x - BOX_PADDING / 2, y - BOX_PADDING / 2, BOX_PADDING, BOX_PADDING, focus_page),
            "reading_order_sections",
        )

        if click_sections:
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
        elif click_sections := focus_document.find(
            Box(x - BOX_PADDING / 2, y - BOX_PADDING / 2, BOX_PADDING, BOX_PADDING, focus_page),
            "tables",
        ):
            section_name = click_sections[0].id
            paragraph = ""
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

        else:
            st.toast("No parsed content found at specified location!")

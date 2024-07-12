import pandas as pd
import spacy_streamlit
from streamlit_dimensions import st_dimensions
from streamlit_image_coordinates import streamlit_image_coordinates


from papermage import Box, Entity
from papermage_components.constants import (
    HIGHLIGHT_COLORS,
    HIGHLIGHT_TYPES,
    MAT_IE_COLORS,
)

from interface_utils import *
from papermage_components.utils import (
    get_table_image,
    visualize_highlights,
    visualize_tagged_entities,
)


st.set_page_config(layout="wide")

# CONSTANTS
BOX_PADDING = 0.01

file_options = os.listdir(PARSED_PAPER_FOLDER)
show_model_annotations = {}
model_entity_type_filter = {}


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


with st.sidebar:
    st.write("Select a parsed file whose results to display")
    focus_file = st.session_state.get("focus_document")
    file_selector = st.selectbox(
        "Parsed file",
        options=file_options,
        index=file_options.index(focus_file) if focus_file else 0,
    )
    st.session_state["focus_document"] = file_selector
    focus_document = load_document(file_selector)

    st.write("Show tagging results on text from:")
    show_user_highlights = st.toggle("User Highlights", value=False)
    for model_name in infer_tagging_models(focus_document):
        show_model_annotations[model_name] = st.toggle(model_name, value=True)
        if show_model_annotations[model_name]:
            model_entity_types = get_entity_types([model_name])
            model_entity_type_filter[model_name] = st.multiselect(
                "Entity types to display:",
                options=model_entity_types,
                default=model_entity_types,
                key=f"entity_type_select_{model_name}",
            )


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
            if show_user_highlights:
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

            for model_name, show_model in show_model_annotations.items():
                if not show_model:
                    continue
                st.write(f"### Annotations from {model_name}")
                spacy_doc = visualize_tagged_entities(
                    paragraph_entity=entity,
                    spacy_pipeline=get_spacy_pipeline(),
                    model_name=model_name,
                    allowed_entity_types=model_entity_type_filter[model_name],
                )
                spacy_streamlit.visualize_ner(
                    spacy_doc,
                    labels=get_entity_types([model_name]),
                    show_table=False,
                    # displacy_options={"colors": MAT_IE_COLORS},
                    key=f"highlights_{model_name}",
                    title=None,
                )

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
    st.write(
        "Click a section of text to view the annotations on it. Light blue boxes indicate clickable"
        " areas that have been annotated. Green boxes indicate the current section of focus."
    )
    if isinstance(section_name, str):
        highlighted_image = highlight_section_on_page(
            focus_document, focus_page, section_name, paragraph
        )
    elif isinstance(section_name, int):
        highlighted_image = highlight_table_on_page(
            focus_document, focus_page, [focus_document.tables[section_name]]
        )
    else:
        highlighted_image = plot_selectable_regions(focus_document, focus_page)
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

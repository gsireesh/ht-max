from streamlit_dimensions import st_dimensions
from streamlit_image_coordinates import streamlit_image_coordinates

from papermage import Box
from shared_utils import *

st.set_page_config(layout="wide")

# CONSTANTS
BOX_PADDING = 0.01

file_options = os.listdir(PARSED_PAPER_FOLDER)

with st.sidebar:  # .form("File selector"):
    st.write("Select a pre-parsed file whose results to display")
    file_selector = st.selectbox("Parsed file", options=file_options)
    focus_document = load_document(file_selector)

doc_vis_column_inspect, sections_column_inspect = st.columns([0.4, 0.6])
with doc_vis_column_inspect:
    focus_page = st.slider(
        label="Select the document page to view",
        min_value=1,
        max_value=len(focus_document.pages),
        value=1,
        format="Page %d",
        key="inspection_slider",
    )
    focus_page = focus_page - 1

clicked_section = st.session_state.get("clicked_section_inspection", None)
if (
    clicked_section is None
    or clicked_section[0] != file_selector
    or clicked_section[1] != focus_page
):
    st.session_state["clicked_section_inspection"] = None
    section_name = None
    paragraph = None
else:
    section_name = clicked_section[2]
    paragraph = clicked_section[3]

with sections_column_inspect:
    if section_name is not None:
        section_entities = [
            e
            for e in focus_document.pages[focus_page].reading_order_sections
            if e.metadata["section_name"] == section_name
            and e.metadata["paragraph_reading_order"] == paragraph
        ]
        st.markdown(f"## {section_entities[0].metadata['section_name']}")

        for entity in section_entities:
            st.write(entity.text)

with doc_vis_column_inspect:
    st.write("Click a section of text to view the raw text that was parsed from it:")
    highlighted_image = highlight_section_on_page(
        focus_document, focus_page, section_name, paragraph
    )
    page_width, page_height = highlighted_image.pilimage.size
    ratio = page_height / page_width

    image_width = st_dimensions(key="doc_vis_inspect")["width"]
    image_height = image_width * ratio

    image_coords = streamlit_image_coordinates(
        highlighted_image.pilimage, key="inspect_page_image", width=image_width
    )

    if image_coords is not None:
        x = image_coords["x"] / image_width
        y = image_coords["y"] / image_height
    else:
        x, y = 0, 0

    click_sections_inspect = focus_document.find(
        Box(x - BOX_PADDING / 2, y - BOX_PADDING / 2, BOX_PADDING, BOX_PADDING, focus_page),
        "reading_order_sections",
    )
    if not click_sections_inspect:
        st.toast("No parsed content found at specified location!")
    else:
        section_name = click_sections_inspect[0].metadata["section_name"]
        paragraph = click_sections_inspect[0].metadata["paragraph_reading_order"]
        if st.session_state.get("clicked_section_inspection") != (
            file_selector,
            focus_page,
            section_name,
            paragraph,
        ):
            st.session_state["clicked_section_inspection"] = (
                file_selector,
                focus_page,
                section_name,
                paragraph,
            )
            st.rerun()

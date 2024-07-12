import pandas as pd

from streamlit.column_config import TextColumn

from shared_utils import *
from papermage_components.constants import (
    MAT_IE_TYPES,
)


st.set_page_config(layout="wide")


def get_matie_entities(doc, allowed_sections, allowed_types):
    all_entities = []
    for entity in doc.getattr(f"TAGGED_ENTITIES_MatIE", []):
        if not entity.reading_order_sections:
            continue

        section_name = entity.reading_order_sections[0].metadata["section_name"]
        entity_type = entity.metadata["entity_type"]

        if section_name not in allowed_sections or entity_type not in allowed_type:
            continue

        sentence_context = entity.sentences[0].text
        all_entities.append(
            {
                "entity_type": entity_type,
                "entity_text": entity.text,
                "entity_section": section_name,
                "sentence_context": sentence_context,
                "source_model": "MatIE",
            }
        )
    return all_entities


def get_gpt_entities(doc, allowed_sections, allowed_types):
    all_entities = []
    for section in doc.reading_order_sections:
        section_name = section.metadata["section_name"]
        if section_name not in allowed_sections:
            continue
        if (gpt_entities := section.metadata.get("gpt_recognized_entities")) is None:
            continue
        for entity in gpt_entities:
            entity_type = entity.get("entity_type").replace(" ", "_")
            if entity_type not in allowed_types:
                continue
            all_entities.append(
                {
                    "entity_type": entity_type,
                    "entity_text": entity.get("entity_string"),
                    "sentence_context": entity.get("entity_context"),
                    "entity_section": section_name,
                    "source_model": "GPT-3.5",
                }
            )
    return all_entities


def get_tables(doc, filter_string):
    tables_to_return = []
    for table in doc.tables:
        table_dict = table.metadata["table_dict"]
        filter_match = any([filter_string in header for header in table_dict])

        caption_id = table.metadata.get("caption_id")
        caption = focus_document.captions[caption_id].text if caption_id else ""
        if caption:
            substring_idx = caption.lower().index("table") if "table" in caption.lower() else 0
            caption = caption[substring_idx:]

        filter_match = filter_match or filter_string in caption.lower()
        if not table_dict or not filter_match:
            continue
        tables_to_return.append(table)
    return tables_to_return


file_options = os.listdir(PARSED_PAPER_FOLDER)

with st.sidebar:  # .form("File selector"):
    st.write("Select a pre-parsed file whose results to display")
    focus_file = st.session_state.get("focus_document")
    file_selector = st.selectbox(
        "Parsed file",
        options=file_options,
        index=file_options.index(focus_file) if focus_file else 0,
    )
    st.session_state["focus_document"] = file_selector
    focus_document = load_document(file_selector)


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

    show_matie_entities = st.toggle("Show MatIE Entities")
    show_gpt_entities = st.toggle("Show GPT Entities")

    entities = []
    if show_matie_entities:
        entities = entities + get_matie_entities(
            focus_document, allowed_sections=section_choice, allowed_types=entity_type_choice
        )
    if show_gpt_entities:
        entities = entities + get_gpt_entities(
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
    text_filter = st.text_input(
        "table_filter",
        label_visibility="collapsed",
        placeholder="Filter tables based on captions or column headers:",
    )

    filtered_tables = get_tables(focus_document, text_filter)
    st.write(f"Found {len(filtered_tables)} matching tables")
    for table in filtered_tables:
        with st.container(border=True):
            table_dict = table.metadata["table_dict"]
            st.dataframe(pd.DataFrame(table_dict))
            table_page = table.boxes[0].page
            caption_id = table.metadata.get("caption_id")
            if caption_id:
                caption = focus_document.captions[caption_id].text
                substring_idx = caption.lower().index("table") if "table" in caption.lower() else 0
                st.write(f"**Identified Caption**: {caption[substring_idx:]}")
            st.write(f"From page {table_page + 1}")

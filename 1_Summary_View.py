import pandas as pd

from streamlit.column_config import TextColumn

from shared_utils import *
from papermage_components.constants import (
    MAT_IE_TYPES,
)


st.set_page_config(layout="wide")


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


file_options = os.listdir(PARSED_PAPER_FOLDER)

with st.sidebar:  # .form("File selector"):
    st.write("Select a pre-parsed file whose results to display")
    file_selector = st.selectbox("Parsed file", options=file_options)
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


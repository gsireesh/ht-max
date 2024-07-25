import pandas as pd
from streamlit.column_config import TextColumn

from interface_utils import *
from interface_utils import get_entity_types, infer_token_predictors


st.set_page_config(layout="wide")


file_options = os.listdir(PARSED_PAPER_FOLDER)
show_model_annotations = {}
model_entity_type_filter = {}


def get_tagged_entities(doc, model_name, allowed_sections, allowed_types):
    all_entities = []
    for section in doc.reading_order_sections:
        if section.metadata["section_name"] not in allowed_sections:
            continue
        for entity in getattr(section, f"TAGGED_ENTITIES_{model_name}", []):
            if entity.metadata["entity_type"] not in allowed_types:
                continue
            sentence_context = entity.sentences[0].text
            all_entities.append(
                {
                    "entity_type": entity.metadata["entity_type"],
                    "entity_text": entity.text,
                    "entity_section": section.metadata["section_name"],
                    "sentence_context": sentence_context,
                    "source_model": model_name,
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

    st.write("Show predicted results from:")

    for model_name in infer_token_predictors(focus_document):
        show_model_annotations[model_name] = st.toggle(model_name, value=True)
        if show_model_annotations[model_name]:
            model_entity_types = get_entity_types([model_name])
            model_entity_type_filter[model_name] = st.multiselect(
                "Entity types to display:",
                options=model_entity_types,
                default=model_entity_types,
                key=f"entity_type_select_{model_name}",
            )


entities_column, table_column = st.columns([0.5, 0.5])
with entities_column:
    st.write("## Tagged Entities")
    all_sections = {e.metadata["section_name"] for e in focus_document.reading_order_sections}
    all_entity_types = get_entity_types(
        [model_name for model in show_model_annotations if show_model_annotations[model]]
    )

    section_choice = st.multiselect(
        label="Choose sections from which to display entities",
        options=all_sections,
        default=all_sections,
    )

    entities = []
    for predictor_name, show in show_model_annotations.items():
        if show:
            entities = entities + get_tagged_entities(
                focus_document,
                predictor_name,
                allowed_sections=section_choice,
                allowed_types=model_entity_type_filter[predictor_name],
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

import pandas as pd
from streamlit.column_config import TextColumn

from interface_utils import *
from interface_utils import get_entity_types, infer_token_predictors
from papermage_components.matie_heuristics import (
    get_most_common_materials,
    create_document_graph,
    get_property_table,
    get_synthesis_method_table,
)
from papermage_components.utils import visualize_table_with_boxes

st.set_page_config(layout="wide")


file_options = os.listdir(PARSED_PAPER_FOLDER)
show_text_annotations_from = {}
show_image_annotations_from = {}
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


def get_processed_images(doc, model_name):
    layer = getattr(doc, f"TAGGED_IMAGE_{model_name}")
    return layer.entities


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
        show_text_annotations_from[model_name] = st.toggle(model_name, value=True)
        if show_text_annotations_from[model_name]:
            model_entity_types = get_entity_types(model_name, focus_document)
            model_entity_type_filter[model_name] = st.multiselect(
                "Entity types to display:",
                options=model_entity_types,
                default=model_entity_types,
                key=f"entity_type_select_{model_name}",
            )

    st.divider()
    for model_name in infer_image_predictors(focus_document):
        show_image_annotations_from[model_name] = st.toggle(model_name, value=True)


entities_column, table_column = st.columns([0.5, 0.5])
with entities_column:
    existing_tab, heuristics_tab = st.tabs(["Entity Overview", "Material Heuristics"])
    with existing_tab:
        st.write("## Tagged Entities")
        all_sections = {e.metadata["section_name"] for e in focus_document.reading_order_sections}

        section_choice = st.multiselect(
            label="Choose sections from which to display entities",
            options=all_sections,
            default=all_sections,
        )

        entities = []
        for predictor_name, show in show_text_annotations_from.items():
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

    with heuristics_tab:
        doc_graph = create_document_graph(focus_document, str(focus_document.symbols[:50]))

        st.write("## Most common materials")
        most_frequent_materials = get_most_common_materials(focus_document.TAGGED_ENTITIES_MatIE)
        cols = st.columns(len(most_frequent_materials))
        for i, ((material, count), col) in enumerate(zip(most_frequent_materials, cols)):
            st.metric(f"#{i + 1}", material, f"{count} mentions")

        st.write("Discovered Property Table")
        property_table = get_property_table(doc_graph)
        st.dataframe(property_table)

        st.write("Discovered Synthesis Table")
        synthesis_table = get_synthesis_method_table(doc_graph)
        st.dataframe(synthesis_table)


with table_column:
    st.write("## Processed Images")
    # text_filter = st.text_input(
    #     "table_filter",
    #     label_visibility="collapsed",
    #     placeholder="Filter tables based on captions or column headers:",
    # )

    for model_name, show_model in show_image_annotations_from.items():
        if show_model:
            result_entities = get_processed_images(focus_document, model_name)
            display_format = st.selectbox(
                "Visualization Format",
                result_entities[0].metadata.keys(),
                format_func=lambda x: x.replace("_", " ").title(),
                index=1,
            )
            for entity in result_entities:
                if not entity.metadata[display_format]:
                    continue
                with st.container(border=True):
                    if display_format == "raw_prediction" and entity.metadata.get("raw_prediction"):
                        st.write(entity.metadata["raw_prediction"])
                    elif display_format == "predicted_dict" and entity.metadata.get(
                        "predicted_dict"
                    ):
                        st.write(pd.DataFrame(entity.metadata["predicted_dict"]))
                    elif display_format == "predicted_boxes" and entity.metadata.get(
                        "predicted_boxes"
                    ):
                        table_visualized = visualize_table_with_boxes(
                            entity,
                            entity.metadata.get("predicted_boxes"),
                            focus_document,
                            False,
                        )
                        st.write(table_visualized)
                    st.write(f"**Predicted by:** {model_name}")
                    st.write(f"**Location**: Page {entity.boxes[0].page}")
                    if caption := entity.metadata.get("predicted_caption"):
                        st.write(f"**Best guess caption:** {caption}")

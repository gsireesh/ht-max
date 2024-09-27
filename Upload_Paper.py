from dataclasses import dataclass
from io import BytesIO
import json
import os
import subprocess
from typing import Any
import warnings

from huggingface_hub import HfApi
from papermage.magelib import (
    BlocksFieldName,
    Box,
    Document,
    Metadata,
    SentencesFieldName,
    TokensFieldName,
    WordsFieldName,
)
from papermage.predictors.word_predictors import make_text
from papermage.utils.annotate import group_by
import streamlit as st
from streamlit_extras.st_keyup import st_keyup
from streamlit_extras.stylable_container import stylable_container

from app_config import app_config as config
from papermage_components.hf_token_classification_predictor import HfTokenClassificationPredictor
from papermage_components.llm_completion_predictor import (
    AVAILABLE_LLMS,
    DEFAULT_MATERIALS_PROMPT,
    LiteLlmCompletionPredictor,
    get_prompt_generator,
    check_valid_key,
)
from papermage_components.materials_recipe import MaterialsRecipe, VILA_LABELS_MAP
from interface_utils import CUSTOM_MODELS_KEY, PARSED_PAPER_FOLDER, EXPECTED_PARSE_LAYERS
from local_model_config import AVAILABLE_LOCAL_MODELS


st.set_page_config(layout="wide")

## resources

UPLOADED_PDF_PATH = config["uploaded_pdf_path"]

pagelink_style = """a[data-testid="stPageLink-NavLink"]
{
    border-radius: 0.5em;
    border: 1px solid rgba(49, 51, 63, 0.2);
    line-height: 2.0;
    justify-content: center;
    font-weight: 400;
    }
}

a:hover {
        border-color: rgb(255,75,75);
        color: rgb(255,75,75);
}
"""


@dataclass
class CustomModelInfo:
    token_predictors: set[str]
    llm_predictors: set[LiteLlmCompletionPredictor]
    local_predictors: set[str]

    def is_empty(self):
        return not (bool(self.token_predictors) or bool(self.llm_predictors))


def reset_custom_models():
    st.session_state[CUSTOM_MODELS_KEY] = CustomModelInfo(set(), set(), set())


if CUSTOM_MODELS_KEY not in st.session_state:
    st.session_state[CUSTOM_MODELS_KEY] = CustomModelInfo(set(), set(), set())


@st.cache_resource
def get_recipe():
    recipe = MaterialsRecipe(
        # matIE_directory="/Users/sireeshgururaja/src/MatIE",
        grobid_server_url=config["grobid_url"],
        gpu_id="mps",
        dpi=150,
    )
    return recipe


@st.cache_resource
def get_hf_tagger(model_name):
    return HfTokenClassificationPredictor(model_name, device="cpu")


def validate_and_add_llm(model_name: str, api_key: str, prompt_string: str) -> None:
    llm_predictor = LiteLlmCompletionPredictor(
        model_name=model_name,
        api_key=api_key,
        prompt_generator_function=get_prompt_generator(prompt_string),
    )

    validation_result = llm_predictor.validate()
    if validation_result.is_valid:
        st.session_state[CUSTOM_MODELS_KEY].llm_predictors.add(llm_predictor)
    else:
        st.error(validation_result.failure_message)


def process_paper(uploaded_paper: BytesIO, container: Any) -> None:
    with container:
        if uploaded_paper is not None:
            bytes_data = uploaded_paper.read()
            paper_filename = os.path.join(UPLOADED_PDF_PATH, uploaded_paper.name)
            with open(paper_filename, "wb") as f:
                f.write(bytes_data)
            recipe = get_recipe()

            try:
                parsed_paper = parse_pdf(paper_filename, recipe)
            except Exception as e:
                st.error(
                    "Your paper failed to parse. Please contact the developers,"
                    " or try a different paper."
                )

            if recipe.matIE_predictor is not None:
                with st.status("Running MatIE Annotation...") as model_status:
                    try:
                        doc = parsed_paper
                        matIE_entities = recipe.matIE_predictor.predict(doc=doc)
                        doc.annotate_layer(
                            name=recipe.matIE_predictor.preferred_layer_name,
                            entities=matIE_entities,
                        )
                        if "entity_types" not in doc.metadata:
                            doc.metadata["entity_types"] = {}
                        doc.metadata["entity_types"][
                            recipe.matIE_predictor.predictor_identifier
                        ] = recipe.matIE_predictor.entity_types
                    except subprocess.CalledProcessError as e:
                        st.write("MatIE failed to run the delegate process.")
                        st.write(f"Error code {e.returncode}; stderr: {e.stderr}")
                    except Exception as e:
                        st.write(e)
                        model_status.update(state="error")

            for local_predictor in st.session_state[CUSTOM_MODELS_KEY].local_predictors:
                with st.status(f"Running model {local_predictor}") as model_status:
                    try:
                        predictor = AVAILABLE_LOCAL_MODELS[local_predictor].get_model()
                        model_entities = predictor.predict(parsed_paper)
                        parsed_paper.annotate_layer(predictor.preferred_layer_name, model_entities)

                        if getattr(predictor, "entity_types", None):
                            if "entity_types" not in parsed_paper.metadata:
                                parsed_paper.metadata["entity_types"] = {}
                            parsed_paper.metadata["entity_types"][
                                predictor.predictor_identifier
                            ] = predictor.entity_types
                    except Exception as e:
                        st.write(e)
                        model_status.update(state="error")

            for token_predictor in st.session_state[CUSTOM_MODELS_KEY].token_predictors:
                with st.status(f"Running model {token_predictor}") as model_status:
                    try:
                        predictor = get_hf_tagger(token_predictor)
                        model_entities = predictor.predict(parsed_paper)
                        parsed_paper.annotate_layer(predictor.preferred_layer_name, model_entities)
                        if "entity_types" not in parsed_paper.metadata:
                            parsed_paper.metadata["entity_types"] = {}
                        parsed_paper.metadata["entity_types"][
                            predictor.predictor_identifier
                        ] = predictor.entity_types
                    except Exception as e:
                        st.write(e)
                        model_status.update(state="error")

            for llm_predictor in st.session_state[CUSTOM_MODELS_KEY].llm_predictors:
                with st.status(f"Generating responses from {llm_predictor.predictor_identifier}"):
                    try:
                        model_entities = llm_predictor.predict(parsed_paper)
                        parsed_paper.annotate_layer(
                            llm_predictor.preferred_layer_name, model_entities
                        )
                    except Exception as e:
                        st.write(e)
                        model_status.update(state="error")

            with open(
                os.path.join(PARSED_PAPER_FOLDER, uploaded_paper.name.replace("pdf", "json")), "w"
            ) as f:
                json.dump(parsed_paper.to_json(), f, indent=4)

            st.session_state["focus_document"] = uploaded_paper.name.replace("pdf", "json")
            st.write(
                "Done processing paper! Expand any failed sections above to see the stack trace."
            )
            st.balloons()

            with stylable_container(key="page_link_style", css_styles=pagelink_style):
                st.page_link(
                    "pages/1_Summary_View.py",
                    label="View Summary of Annotations",
                    icon="🔍",
                )


def parse_pdf(pdf, _recipe) -> Document:
    parsed_doc_filename = os.path.join(
        PARSED_PAPER_FOLDER, os.path.basename(pdf.replace("pdf", "json"))
    )
    st.write(parsed_doc_filename)
    if os.path.exists(parsed_doc_filename):
        with st.status("Paper has already been parsed! Using cached version...") as status:
            try:
                with open(parsed_doc_filename) as f:
                    doc = Document.from_json(json.load(f))
                    for layer in doc.layers:
                        if layer not in EXPECTED_PARSE_LAYERS:
                            doc.remove_layer(layer)
                    return doc
            except Exception as e:
                status.update(
                    state="error",
                    label="Failed to parse cached version. Parsing from scratch.",
                )
                raise e

    with st.status("Parsing PDF...") as status:
        try:
            doc = _recipe.pdfplumber_parser.parse(input_pdf_path=pdf)
        except Exception as e:
            status.update(state="error")
            st.write(e)
            raise e

    with st.status("Getting sections in reading order...") as status:
        try:
            doc = _recipe.grobid_order_parser.parse(
                pdf,
                doc,
            )
        except Exception as e:
            status.update(state="error")
            st.write(e)
            raise e

    with st.status("Rasterizing Document...") as status:
        try:
            images = _recipe.rasterizer.rasterize(input_pdf_path=pdf, dpi=_recipe.dpi)
            doc.annotate_images(images=list(images))
            _recipe.rasterizer.attach_images(images=images, doc=doc)
        except Exception as e:
            status.update(state="error")
            st.write(e)
            raise e

    with st.status("Predicting words...") as status:
        try:
            words = _recipe.word_predictor.predict(doc=doc)
            doc.annotate_layer(name=WordsFieldName, entities=words)
        except Exception as e:
            status.update(state="error")
            st.write(e)
            raise e

    with st.status("Predicting sentences...") as status:
        try:
            sentences = _recipe.sent_predictor.predict(doc=doc)
            doc.annotate_layer(name=SentencesFieldName, entities=sentences)
        except Exception as e:
            status.update(state="error")
            st.write(e)
            raise e

    with st.status("Predicting blocks...") as status:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                blocks = _recipe.publaynet_block_predictor.predict(doc=doc)
            doc.annotate_layer(name=BlocksFieldName, entities=blocks)
        except Exception as e:
            status.update(state="error")
            st.write(e)
            raise e

    with st.status("Predicting vila...") as status:
        try:
            vila_entities = _recipe.ivila_predictor.predict(doc=doc)
            doc.annotate_layer(name="vila_entities", entities=vila_entities)
            for entity in vila_entities:
                entity.boxes = [
                    Box.create_enclosing_box(
                        [
                            b
                            for t in doc.intersect_by_span(entity, name=TokensFieldName)
                            for b in t.boxes
                        ]
                    )
                ]
                entity.text = make_text(entity=entity, document=doc)
            preds = group_by(
                entities=vila_entities, metadata_field="label", metadata_values_map=VILA_LABELS_MAP
            )
            doc.annotate(*preds)
        except Exception as e:
            status.update(state="error")
            st.write(e)
            raise e

    return doc


st.title("Welcome to Collage!")

col1, col2 = st.columns([0.6, 0.4])
with col1:
    st.write("## 1. Customize the pipeline that runs on your paper")
    with st.status("Basic Processing"):
        st.checkbox("Parse + Rasterize PDF", value=True, disabled=True)
        st.checkbox("Get sections in reading order", value=True, disabled=True)
        st.checkbox("Predict words", value=True, disabled=True)
        st.checkbox("Predict sentences", value=True, disabled=True)
        st.checkbox("Predict blocks", value=True, disabled=True)
        st.checkbox("Predict VILA", value=True, disabled=True)

    with st.status("Available custom models:", expanded=True):
        for model_info in AVAILABLE_LOCAL_MODELS.values():
            model_name = model_info.model_name
            use_model = st.checkbox(model_name, value=model_info.use_by_default)
            if use_model:
                st.session_state[CUSTOM_MODELS_KEY].local_predictors.add(model_name)
            else:
                st.session_state[CUSTOM_MODELS_KEY].local_predictors.discard(model_name)

    if not st.session_state.get(CUSTOM_MODELS_KEY).is_empty():
        with st.status("Additional Models:", expanded=True):
            if st.session_state[CUSTOM_MODELS_KEY].token_predictors:
                st.write("**HuggingFace Models:**")
                for model_name in st.session_state[CUSTOM_MODELS_KEY].token_predictors:
                    st.write(model_name)

            if st.session_state[CUSTOM_MODELS_KEY].llm_predictors:
                st.write("**Custom LLMs:**")
                for model in st.session_state[CUSTOM_MODELS_KEY].llm_predictors:
                    st.write(model.predictor_identifier)

            st.button("Clear all", on_click=reset_custom_models)

    st.divider()
    hf_tab, llm_tab = st.tabs(["Add HuggingFace Token Classifiers", "Add an LLM Predictor"])
    with hf_tab, st.container(border=True):
        st.write("### Add a HuggingFace Token Classification Model")
        model_name_input = st_keyup(label="Model Name Filter", debounce=500)
        hf_api = HfApi()

        results = hf_api.list_models(
            model_name=model_name_input,
            pipeline_tag="token-classification",
            library="transformers",
            language="en",
            sort="downloads",
            direction=-1,
            limit=5,
        )
        for result in results:
            model_name = result.id
            model_name_col, use_model_col = st.columns([0.7, 0.3])
            model_name_col.write(f"{model_name}\n(⬇️ {result.downloads})")
            use_model_col.button(
                "Use this model",
                type="primary",
                key=f"use_{model_name}",
                on_click=lambda custom_model_name: st.session_state[
                    CUSTOM_MODELS_KEY
                ].token_predictors.add(custom_model_name),
                kwargs={"custom_model_name": model_name},
            )

    with llm_tab, st.container(border=True):
        model_name = st.selectbox(label="Select model:", options=AVAILABLE_LLMS, index=6)
        api_key = st_keyup(
            "API Key:", value=config["llm_api_keys"].get(model_name, ""), debounce=500
        )

        if check_valid_key(model=model_name, api_key=api_key):
            st.write("✅ Valid API Key.")
        else:
            st.write("🚩 Invalid API Key!")

        with st.expander("Customize prompt:"):
            prompt_string = st.text_area(
                label="prompt_text",
                value=DEFAULT_MATERIALS_PROMPT,
                label_visibility="collapsed",
                height=300,
            )
        st.button(
            "Add LLM",
            type="primary",
            on_click=validate_and_add_llm,
            kwargs={"model_name": model_name, "api_key": api_key, "prompt_string": prompt_string},
        )


with col2:
    st.write("## 2. Upload a file to process")
    uploaded_file = st.file_uploader(
        "Upload a paper to process.", type="pdf", accept_multiple_files=False
    )
    st.button(
        "Process uploaded paper",
        on_click=process_paper,
        kwargs={"uploaded_paper": uploaded_file, "container": col2},
    )

import json
import os
import warnings

from huggingface_hub import HfApi
from papermage.magelib import (
    BlocksFieldName,
    Box,
    Document,
    SentencesFieldName,
    TokensFieldName,
    WordsFieldName,
)
from papermage.predictors.word_predictors import make_text
from papermage.utils.annotate import group_by
import streamlit as st
from streamlit_extras.st_keyup import st_keyup
from streamlit_extras.stylable_container import stylable_container

from papermage_components.hf_token_classification_predictor import HfTokenClassificationPredictor
from papermage_components.materials_recipe import MaterialsRecipe, VILA_LABELS_MAP
from shared_utils import CUSTOM_MODELS_KEY, PARSED_PAPER_FOLDER


st.set_page_config(layout="wide")


## resources

UPLOADED_PDF_PATH = "data/uploaded_papers"

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


def reset_custom_models():
    st.session_state[CUSTOM_MODELS_KEY] = set()


if CUSTOM_MODELS_KEY not in st.session_state:
    st.session_state[CUSTOM_MODELS_KEY] = set()


@st.cache_resource
def get_recipe():
    recipe = MaterialsRecipe(
        # matIE_directory="/Users/sireeshgururaja/src/MatIE",
        grobid_server_url="http://windhoek.sp.cs.cmu.edu:8070",
        gpu_id="mps",
    )
    return recipe


@st.cache_resource
def get_hf_tagger(model_name):
    return HfTokenClassificationPredictor(model_name, device="cpu")


def process_paper(uploaded_paper, container):
    with container:
        if uploaded_paper is not None:
            bytes_data = uploaded_paper.read()
            paper_filename = os.path.join(UPLOADED_PDF_PATH, uploaded_paper.name)
            with open(paper_filename, "wb") as f:
                f.write(bytes_data)
            recipe = get_recipe()

            parsed_paper = parse_pdf(paper_filename, recipe)

            for model_name in set(st.session_state[CUSTOM_MODELS_KEY]):
                with st.status(f"Running model {model_name}") as model_status:
                    try:
                        predictor = get_hf_tagger(model_name)
                        model_entities = predictor.predict(parsed_paper)
                        parsed_paper.annotate_layer(f"ENTITIES_{model_name}", model_entities)
                    except Exception as e:
                        st.write(e)
                        model_status.update("error")

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

    with st.status("Parsing PDF...") as status:
        try:
            doc = _recipe.pdfplumber_parser.parse(input_pdf_path=pdf)
        except Exception as e:
            status.update(state="error")
            st.write(e)

    with st.status("Getting sections in reading order...") as status:
        try:
            doc = _recipe.grobid_order_parser.parse(
                pdf,
                doc,
            )
        except Exception as e:
            status.update(state="error")
            st.write(e)

    with st.status("Rasterizing Document...") as status:
        try:
            images = _recipe.rasterizer.rasterize(input_pdf_path=pdf, dpi=_recipe.dpi)
            doc.annotate_images(images=list(images))
            _recipe.rasterizer.attach_images(images=images, doc=doc)
        except Exception as e:
            st.write(e)

    with st.status("Predicting words...") as status:
        try:
            words = _recipe.word_predictor.predict(doc=doc)
            doc.annotate_layer(name=WordsFieldName, entities=words)
        except Exception as e:
            status.update(state="error")
            st.write(e)

    with st.status("Predicting sentences...") as status:
        try:
            sentences = _recipe.sent_predictor.predict(doc=doc)
            doc.annotate_layer(name=SentencesFieldName, entities=sentences)
        except Exception as e:
            status.update(state="error")
            st.write(e)

    with st.status("Predicting blocks...") as status:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                blocks = _recipe.publaynet_block_predictor.predict(doc=doc)
            doc.annotate_layer(name=BlocksFieldName, entities=blocks)
        except Exception as e:
            status.update(state="error")
            st.write(e)

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

    return doc


st.title("Welcome to Collage!")

col1, col2 = st.columns([0.4, 0.6])
with col1:
    st.write("## 1. Customize the pipeline that runs on your paper")
    with st.status("Basic Processing"):
        st.checkbox("Parse + Rasterize PDF", value=True, disabled=True)
        st.checkbox("Get sections in reading order", value=True, disabled=True)
        st.checkbox("Predict words", value=True, disabled=True)
        st.checkbox("Predict sentences", value=True, disabled=True)
        st.checkbox("Predict blocks", value=True, disabled=True)
        st.checkbox("Predict VILA", value=True, disabled=True)

    if st.session_state.get(CUSTOM_MODELS_KEY):
        with st.status("Additional Models:", expanded=True):
            for model_name in st.session_state[CUSTOM_MODELS_KEY]:
                st.write(model_name)

            st.button("Clear all", on_click=reset_custom_models)

    with st.container(border=True):
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
                on_click=lambda custom_model_name: st.session_state[CUSTOM_MODELS_KEY].add(
                    custom_model_name
                ),
                kwargs={"custom_model_name": model_name},
            )


with col2:
    with st.form("file_upload_form"):
        st.write("## 2. Upload a file to process")
        uploaded_file = st.file_uploader(
            "Upload a paper to process.", type="pdf", accept_multiple_files=False
        )
        st.form_submit_button(
            "Process uploaded paper",
            on_click=process_paper,
            kwargs={"uploaded_paper": uploaded_file, "container": col2},
        )

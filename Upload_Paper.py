import os
import re
import warnings

from papermage.magelib import (
    BlocksFieldName,
    Box,
    SentencesFieldName,
    TokensFieldName,
    WordsFieldName,
)
from papermage.predictors.word_predictors import make_text
from papermage.utils.annotate import group_by
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from papermage_components.materials_recipe import MaterialsRecipe, VILA_LABELS_MAP
from shared_utils import *

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


@st.cache_resource
def get_recipe():
    recipe = MaterialsRecipe(
        # matIE_directory="/Users/sireeshgururaja/src/MatIE",
        grobid_server_url="http://windhoek.sp.cs.cmu.edu:8070",
    )
    return recipe


def parse_pdf(pdf, recipe):

    with st.status("Parsing PDF...") as status:
        try:
            doc = recipe.pdfplumber_parser.parse(input_pdf_path=pdf)
        except Exception as e:
            status.update(state="error")
            st.write(e)

    with st.status("Getting sections in reading order...") as status:
        try:
            doc = recipe.grobid_order_parser.parse(
                pdf,
                doc,
            )
        except Exception as e:
            status.update(state="error")
            st.write(e)

    with st.status("Rasterizing Document...") as status:
        try:
            images = recipe.rasterizer.rasterize(input_pdf_path=pdf, dpi=recipe.dpi)
            doc.annotate_images(images=list(images))
            recipe.rasterizer.attach_images(images=images, doc=doc)
        except Exception as e:
            st.write(e)

    with st.status("Predicting words...") as status:
        try:
            words = recipe.word_predictor.predict(doc=doc)
            doc.annotate_layer(name=WordsFieldName, entities=words)
        except Exception as e:
            status.update(state="error")
            st.write(e)

    with st.status("Predicting sentences...") as status:
        try:
            sentences = recipe.sent_predictor.predict(doc=doc)
            doc.annotate_layer(name=SentencesFieldName, entities=sentences)
        except Exception as e:
            status.update(state="error")
            st.write(e)

    if recipe.matIE_predictor is not None:
        with st.status("Predicting MatIE Entities...") as status:
            try:
                matIE_entities = recipe.matIE_predictor.predict(doc=doc)
                doc.annotate(matIE_entities)
            except Exception as e:
                status.update(state="error")
                st.write(e)

    with st.status("Predicting blocks...") as status:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                blocks = recipe.publaynet_block_predictor.predict(doc=doc)
            doc.annotate_layer(name=BlocksFieldName, entities=blocks)
        except Exception as e:
            status.update(state="error")
            st.write(e)

    with st.status("Predicting vila...") as status:
        try:
            vila_entities = recipe.ivila_predictor.predict(doc=doc)
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

    with st.status("Predicting table structure...") as status:
        try:
            recipe.table_structure_predictor.predict(doc)
        except Exception as e:
            status.update(state="error")
            st.write(e)

    return doc


st.title("Welcome to Collage!")

col1, col2 = st.columns([0.4, 0.6])
with col1:
    with st.form("file_upload_form"):
        uploaded_file = st.file_uploader(
            "Upload a paper to get started.", type="pdf", accept_multiple_files=False
        )
        st.form_submit_button("Process uploaded paper")

with col2:
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        paper_filename = os.path.join(UPLOADED_PDF_PATH, uploaded_file.name)
        with open(paper_filename, "wb") as f:
            f.write(bytes_data)
        recipe = get_recipe()
        parsed_paper = parse_pdf(paper_filename, recipe)
        with open(
            os.path.join(PARSED_PAPER_FOLDER, uploaded_file.name.replace("pdf", "json")), "w"
        ) as f:
            json.dump(parsed_paper.to_json(), f, indent=4)

        st.session_state["focus_document"] = uploaded_file.name.replace("pdf", "json")
        st.write("Done processing paper! Expand any failed sections above to see the stack trace.")
        st.balloons()

        with stylable_container(key="page_link_style", css_styles=pagelink_style):
            st.page_link(
                "pages/1_Summary_View.py",
                label="View Summary of Annotations",
                icon="🔍",
            )

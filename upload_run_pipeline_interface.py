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

from papermage_components.materials_recipe import MaterialsRecipe, VILA_LABELS_MAP
from shared_utils import *

## resources

UPLOADED_PDF_PATH = "data/uploaded_papers"


@st.cache_resource
def get_recipe():
    recipe = MaterialsRecipe(
        matIE_directory="/Users/sireeshgururaja/src/MatIE",
        grobid_server_url="http://windhoek.sp.cs.cmu.edu:8070",
    )
    return recipe


st.title("Collage.")


def parse_pdf(pdf, recipe):

    original_message = "Parsing PDF..."
    output_container = st.empty()
    output_container.write(original_message)
    try:
        doc = recipe.pdfplumber_parser.parse(input_pdf_path=pdf)
        output_container.write(original_message + "✅")
    except Exception as e:
        output_container.write(original_message + "🚩")
        with st.expander("Show stack trace"):
            st.write(e)

    original_message = "Getting sections in reading order..."
    output_container = st.empty()
    output_container.write(original_message)
    try:
        doc = recipe.grobid_order_parser.parse(
            pdf,
            doc,
        )
        output_container.write(original_message + "✅")
    except Exception as e:
        output_container.write(original_message + "🚩")
        with st.expander("Show stack trace"):
            st.write(e)

    original_message = "Rasterizing Document..."
    output_container = st.empty()
    output_container.write(original_message)
    try:
        images = recipe.rasterizer.rasterize(input_pdf_path=pdf, dpi=recipe.dpi)
        doc.annotate_images(images=list(images))
        recipe.rasterizer.attach_images(images=images, doc=doc)
        output_container.write(original_message + "✅")
    except Exception as e:
        output_container.write(original_message + "🚩")
        with st.expander("Show stack trace"):
            st.write(e)

    original_message = "Predicting words..."
    output_container = st.empty()
    output_container.write(original_message)
    try:
        words = recipe.word_predictor.predict(doc=doc)
        doc.annotate_layer(name=WordsFieldName, entities=words)
        output_container.write(original_message + "✅")
    except Exception as e:
        output_container.write(original_message + "🚩")
        with st.expander("Show stack trace"):
            st.write(e)

    original_message = "Predicting sentences..."
    output_container = st.empty()
    output_container.write(original_message)
    try:
        sentences = recipe.sent_predictor.predict(doc=doc)
        doc.annotate_layer(name=SentencesFieldName, entities=sentences)
        output_container.write(original_message + "✅")
    except Exception as e:
        output_container.write(original_message + "🚩")
        with st.expander("Show stack trace"):
            st.write(e)

    if recipe.matIE_predictor is not None:
        original_message = "Predicting MatIE Entities..."
        output_container = st.empty()
        output_container.write(original_message)
        try:
            matIE_entities = recipe.matIE_predictor.predict(doc=doc)
            doc.annotate(matIE_entities)
            output_container.write(original_message + "✅")
        except Exception as e:
            output_container.write(original_message + "🚩")
            with st.expander("Show stack trace"):
                st.write(e)

    original_message = "Predicting blocks..."
    output_container = st.empty()
    output_container.write(original_message)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blocks = recipe.publaynet_block_predictor.predict(doc=doc)
        doc.annotate_layer(name=BlocksFieldName, entities=blocks)
        output_container.write(original_message + "✅")
    except Exception as e:
        output_container.write(original_message + "🚩")
        with st.expander("Show stack trace"):
            st.write(e)

    original_message = "Predicting vila..."
    output_container = st.empty()
    output_container.write(original_message)
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
        output_container.write(original_message + "✅")
    except Exception as e:
        output_container.write(original_message + "🚩")
        with st.expander("Show stack trace"):
            st.write(e)

    original_message = "Predicting table structure..."
    output_container = st.empty()
    output_container.write(original_message)
    try:
        recipe.table_structure_predictor.predict(doc)
        output_container.write(original_message + "✅")
    except Exception as e:
        output_container.write(original_message + "🚩")
        with st.expander("Show stack trace"):
            st.write(e)

    return doc


with st.sidebar:
    with st.form("file_upload_form"):
        uploaded_file = st.file_uploader(
            "Upload papers to get started.", type="pdf", accept_multiple_files=False
        )
        st.form_submit_button("Process uploaded papers")

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
    st.write("Done processing paper!")
    st.balloons()

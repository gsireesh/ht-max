"""
Adapted from
@kylel

"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Union


from papermage.magelib import (
    AbstractsFieldName,
    AlgorithmsFieldName,
    AuthorsFieldName,
    BibliographiesFieldName,
    BlocksFieldName,
    Box,
    CaptionsFieldName,
    Document,
    EntitiesFieldName,
    Entity,
    EquationsFieldName,
    FiguresFieldName,
    FootersFieldName,
    FootnotesFieldName,
    HeadersFieldName,
    ImagesFieldName,
    KeywordsFieldName,
    ListsFieldName,
    PagesFieldName,
    ParagraphsFieldName,
    RelationsFieldName,
    RowsFieldName,
    SectionsFieldName,
    SentencesFieldName,
    SymbolsFieldName,
    TablesFieldName,
    TitlesFieldName,
    TokensFieldName,
    WordsFieldName,
)
from papermage.parsers.pdfplumber_parser import PDFPlumberParser
from papermage.predictors import (
    HFBIOTaggerPredictor,
    IVILATokenClassificationPredictor,
    LPEffDetFormulaPredictor,
    LPEffDetPubLayNetBlockPredictor,
    PysbdSentencePredictor,
    SVMWordPredictor,
)
from papermage.predictors.word_predictors import make_text
from papermage.rasterizers.rasterizer import PDF2ImageRasterizer
from papermage.recipes.recipe import Recipe
from papermage.utils.annotate import group_by

from papermage_components.scispacy_sentence_predictor import SciSpacySentencePredictor
from papermage_components.matIE_predictor import MatIEPredictor
from papermage_components.reading_order_parser import GrobidReadingOrderParser
from papermage_components.highlightParser import FitzHighlightParser
from papermage_components.hf_token_classification_predictor import HfTokenClassificationPredictor
from papermage_components.table_structure_predictor import TableStructurePredictor

VILA_LABELS_MAP = {
    "Title": TitlesFieldName,
    "Paragraph": ParagraphsFieldName,
    "Author": AuthorsFieldName,
    "Abstract": AbstractsFieldName,
    "Keywords": KeywordsFieldName,
    "Section": SectionsFieldName,
    "List": ListsFieldName,
    "Bibliography": BibliographiesFieldName,
    "Equation": EquationsFieldName,
    "Algorithm": AlgorithmsFieldName,
    "Figure": FiguresFieldName,
    "Table": TablesFieldName,
    "Caption": CaptionsFieldName,
    "Header": HeadersFieldName,
    "Footer": FootersFieldName,
    "Footnote": FootnotesFieldName,
}


class MaterialsRecipe(Recipe):
    def __init__(
        self,
        ivila_predictor_path: str = "allenai/ivila-row-layoutlm-finetuned-s2vl-v2",
        bio_roberta_predictor_path: str = "allenai/vila-roberta-large-s2vl-internal",
        svm_word_predictor_path: str = "https://ai2-s2-research-public.s3.us-west-2.amazonaws.com/mmda/models/svm_word_predictor.tar.gz",
        scispacy_model: str = "en_core_sci_md",
        annotated_pdf_directory="data/AM_Creep_Papers_Annotated_2/",
        grobid_server_url: str = "",
        xml_out_dir: str = "data/grobid_xml",
        matIE_directory: str = "",
        gpu_id: str = "0",
        dpi: int = 72,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dpi = dpi

        self.logger.info("Instantiating recipe...")
        self.pdfplumber_parser = PDFPlumberParser()
        self.grobid_order_parser = GrobidReadingOrderParser(
            grobid_server_url, check_server=True, xml_out_dir=xml_out_dir
        )
        self.highlight_parser = FitzHighlightParser(annotated_pdf_directory)
        self.rasterizer = PDF2ImageRasterizer()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.word_predictor = SVMWordPredictor.from_path(svm_word_predictor_path)

        self.publaynet_block_predictor = LPEffDetPubLayNetBlockPredictor.from_pretrained()
        self.ivila_predictor = IVILATokenClassificationPredictor.from_pretrained(
            ivila_predictor_path
        )
        # self.bio_roberta_predictor = HFBIOTaggerPredictor.from_pretrained(
        #     bio_roberta_predictor_path,
        #     entity_name="tokens",
        #     context_name="pages",
        # )
        self.sent_predictor = SciSpacySentencePredictor(
            model_name=scispacy_model,
        )

        if matIE_directory:
            self.matIE_predictor = MatIEPredictor(
                matIE_directory=matIE_directory,
                gpu_id=gpu_id,
            )
        else:
            self.matIE_predictor = None

        self.table_structure_predictor = TableStructurePredictor.from_model_name()

        self.hf_predictor = HfTokenClassificationPredictor(
            model_name="tner/roberta-large-ontonotes5", device=gpu_id
        )

        self.logger.info("Finished instantiating recipe")

    def from_pdf(self, pdf: Path) -> Document:
        self.logger.info("Parsing document...")
        doc = self.pdfplumber_parser.parse(input_pdf_path=pdf)
        self.logger.info("Getting Reading Order Sections...")
        doc = self.grobid_order_parser.parse(
            pdf,
            doc,
        )
        # self.logger.info("Parsing highlights...")
        # doc = self.highlight_parser.parse(pdf, doc)

        self.logger.info("Rasterizing document...")
        images = self.rasterizer.rasterize(input_pdf_path=pdf, dpi=self.dpi)
        doc.annotate_images(images=list(images))
        self.rasterizer.attach_images(images=images, doc=doc)
        return self.from_doc(doc=doc)

    def from_doc(self, doc: Document) -> Document:
        self.logger.info("Predicting words...")
        words = self.word_predictor.predict(doc=doc)
        doc.annotate_layer(name=WordsFieldName, entities=words)

        self.logger.info("Predicting sentences...")
        sentences = self.sent_predictor.predict(doc=doc)
        doc.annotate_layer(name=SentencesFieldName, entities=sentences)

        if self.matIE_predictor is not None:
            self.logger.info("Predicting MatIE Entities...")
            matIE_entities = self.matIE_predictor.predict(doc=doc)
            doc.annotate_layer(
                name=self.matIE_predictor.preferred_layer_name, entities=matIE_entities
            )

        self.logger.info("Predicting blocks...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blocks = self.publaynet_block_predictor.predict(doc=doc)
        doc.annotate_layer(name=BlocksFieldName, entities=blocks)

        self.logger.info("Predicting vila...")
        vila_entities = self.ivila_predictor.predict(doc=doc)
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

        self.logger.info("Predicting table structure")
        self.table_structure_predictor.predict(doc)

        return doc


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=str, help="Path to PDF file.")
    parser.add_argument("--output", type=str, help="Path to output JSON file.")
    args = parser.parse_args()

    recipe = MaterialsRecipe()
    doc = recipe.from_pdf(pdf=args.pdf)
    with open(args.output, "w") as f:
        json.dump(doc.to_json(), f, indent=2)

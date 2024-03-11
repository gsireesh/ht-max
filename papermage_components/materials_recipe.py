"""
Adapted from
@kylel

"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Union

# Import the NER class from NER.py if it's in a separate file
from papermage_components.NER import MatIE


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
        scispacy_model: str = "en_core_sci_scibert",
        dpi: int = 72,
        NER_model_dir: str = "",  # the directory of the NER model
        vocab_dir: str = "",  # the directory of the vocabulary
        output_folder: str = "",  # the directory of the vocabulary
        gpu_id: str = "0",
        decode_script: str = "",  # the decode module
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dpi = dpi

        self.logger.info("Instantiating recipe...")
        self.parser = PDFPlumberParser()
        self.rasterizer = PDF2ImageRasterizer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.word_predictor = SVMWordPredictor.from_path(svm_word_predictor_path)

        # self.publaynet_block_predictor = LPEffDetPubLayNetBlockPredictor.from_pretrained()
        # self.ivila_predictor = IVILATokenClassificationPredictor.from_pretrained(
        #     ivila_predictor_path
        # )
        # self.bio_roberta_predictor = HFBIOTaggerPredictor.from_pretrained(
        #     bio_roberta_predictor_path,
        #     entity_name="tokens",
        #     context_name="pages",
        # )
        self.sent_predictor = SciSpacySentencePredictor(
            model_name=scispacy_model,
        )
        self.matIE_predictor = MatIEPredictor(
            NER_model_dir=NER_model_dir,
            vocab_dir=vocab_dir,
            output_folder=output_folder,
            gpu_id=gpu_id,
            decode_script=decode_script,
        )
        # self.NER = MatIE(NER_model_dir = NER_model_dir, vocab_dir = vocab_dir,
        #         output_folder = output_folder, gpu_id = gpu_id, decode_script = decode_script)

        self.logger.info("Finished instantiating recipe")

    def from_pdf(self, pdf: Path) -> Document:
        self.logger.info("Parsing document...")
        print("pdf", pdf)
        doc = self.parser.parse(input_pdf_path=pdf)

        self.logger.info("Rasterizing document...")
        images = self.rasterizer.rasterize(input_pdf_path=pdf, dpi=self.dpi)
        doc.annotate_images(images=list(images))
        self.rasterizer.attach_images(images=images, doc=doc)
        self.matIE_predictor.curr_file = pdf.split("/")[-1][:-4]

        return self.from_doc(doc=doc)

    def from_doc(self, doc: Document) -> Document:
        self.logger.info("Predicting words...")
        words = self.word_predictor.predict(doc=doc)
        doc.annotate_layer(name=WordsFieldName, entities=words)

        self.logger.info("Predicting sentences...")
        sentences = self.sent_predictor.predict(doc=doc)
        doc.annotate_layer(name=SentencesFieldName, entities=sentences)

        self.logger.info("Predicting MatIE Entities...")
        matIE_entities = self.matIE_predictor.predict(doc=doc)

        # self.logger.info("Predicting blocks...")
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     blocks = self.publaynet_block_predictor.predict(doc=doc)
        # doc.annotate_layer(name=BlocksFieldName, entities=blocks)
        #
        # self.logger.info("Predicting vila...")
        # vila_entities = self.ivila_predictor.predict(doc=doc)
        # doc.annotate_layer(name="vila_entities", entities=vila_entities)
        #
        # for entity in vila_entities:
        #     entity.boxes = [
        #         Box.create_enclosing_box(
        #             [
        #                 b
        #                 for t in doc.intersect_by_span(entity, name=TokensFieldName)
        #                 for b in t.boxes
        #             ]
        #         )
        #     ]
        #     entity.text = make_text(entity=entity, document=doc)
        # preds = group_by(
        #     entities=vila_entities, metadata_field="label", metadata_values_map=VILA_LABELS_MAP
        # )
        # doc.annotate(*preds)
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

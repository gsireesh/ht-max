# Import necessary libraries
import itertools
from typing import List, Tuple
import numpy as np
import spacy
from collections import defaultdict
from papermage.magelib import (
    Document,
    Entity,
    SentencesFieldName,
    TokensFieldName,
    WordsFieldName,
)
from papermage.predictors import BasePredictor
from papermage.utils.merge import cluster_and_merge_neighbor_spans

# Import the NER class from NER.py if it's in a separate file
from papermage_components.NER import MatIE


class MatIEPredictor(BasePredictor):
    def __init__(
        self,
        NER_model_dir="",
        vocab_dir="",
        output_folder="",
        gpu_id="0",
        decode_script="",
    ):

        # Initialize the NER class instance with the output directory
        print(
            "NER_model_dir ", NER_model_dir, "vocab_dir", vocab_dir, "output_folder ", output_folder
        )
        self.ner_instance = MatIE(
            NER_model_dir=NER_model_dir,
            vocab_dir=vocab_dir,
            output_folder=output_folder,
            gpu_id=gpu_id,
            decode_script=decode_script,
        )
        self.curr_file = ""

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [SentencesFieldName, TokensFieldName]

    def _predict(self, doc: Document) -> List[Entity]:
        ie_entities = self.entity_predict(doc, doc.sentences, 10)
        return ie_entities

    def entity_predict(self, doc, sentence_spans: List[Entity], sentence_size=10) -> List[Entity]:
        total_sentences = len(sentence_spans)
        for i in range(0, total_sentences, sentence_size):
            sent_start = sentence_spans[i]
            sent_end = sentence_spans[min(i + sentence_size, total_sentences) - 1]

            subset = sentence_spans[i : i + sentence_size]
            # file_name = doc.TitlesFieldName
            self.ner_instance.generate_txt(self.curr_file, subset, i, i + sentence_size)
            # self.ner_instance.run_matIE()
        return


# Ensure you have the NER class defined in the same file or imported correctly.
# Here you would place the NER class definition if it's not in a separate file.

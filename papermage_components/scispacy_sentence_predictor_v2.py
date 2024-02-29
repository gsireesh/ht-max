# Import necessary libraries
import itertools
from typing import List, Tuple
import numpy as np
import spacy
from collections import defaultdict
from papermage.magelib import (
    Document,
    Entity,
    TokensFieldName,
    WordsFieldName,
)
from papermage.predictors import BasePredictor
from papermage.utils.merge import cluster_and_merge_neighbor_spans

# Import the NER class from NER.py if it's in a separate file
from papermage_components.NER import MatIE



class SciSpacySentencePredictor(BasePredictor):
    def __init__(self, model_name="en_core_sci_scibert", NER_model_dir = "", 
                 vocab_dir = "",output_folder = "", gpu_id = "0", decode_script = ""):
        # Load the SciSpaCy model
        scispacy = spacy.load(model_name)
        self.model = scispacy
        # Initialize the NER class instance with the output directory
        print("NER_model_dir ", NER_model_dir , "vocab_dir", vocab_dir,
              "output_folder ", output_folder)
        self.ner_instance = MatIE(NER_model_dir = NER_model_dir,vocab_dir = vocab_dir,
                                  output_folder = output_folder, 
                                  gpu_id = gpu_id,decode_script = decode_script)
        self.curr_file = ""

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [TokensFieldName]

    def split_token_based_on_sentences_boundary(self, words: List[str]) -> List[Tuple[int, int]]:
        if len(words) == 0:
            return [(0, 0)]
        combined_words = " ".join(words)
        self.combined_words = combined_words
        char2token_mask = np.zeros(len(combined_words), dtype=np.int64)
        acc_word_len = 0
        for idx, word in enumerate(words):
            word_len = len(word) + 1
            char2token_mask[acc_word_len: acc_word_len + word_len] = idx
            acc_word_len += word_len

        doc1 = self.model(combined_words)
        segmented_sentences = list(doc1.sents)
        sent_boundary = [(ele.start_char, ele.end_char) for ele in segmented_sentences]

        split = []
        token_id_start = 0
        for start, end in sent_boundary:
            token_id_end = char2token_mask[start:end].max()
            if end + 1 >= len(char2token_mask) or char2token_mask[end + 1] != token_id_end:
                token_id_end += 1  # Including the end
            split.append((token_id_start, token_id_end))
            token_id_start = token_id_end
        return split

    def _predict(self, doc: Document) -> List[Entity]:
        if hasattr(doc, WordsFieldName):
            words = [word.text for word in getattr(doc, WordsFieldName)]
            attr_name = WordsFieldName
        else:
            words = [token.text for token in doc.tokens]
            attr_name = TokensFieldName

        split = self.split_token_based_on_sentences_boundary(words)
        sentence_spans: List[Entity] = []
        for start, end in split:
            if end - start <= 0:
                continue
            cur_spans = getattr(doc, attr_name)[start:end]
            all_token_spans = list(itertools.chain.from_iterable([ele.spans for ele in cur_spans]))
            results = cluster_and_merge_neighbor_spans(all_token_spans)
            sentence_spans.append(Entity(spans=results.merged))
        self.entity_predict(doc, sentence_spans, 10)
        return sentence_spans

    def entity_predict(self, doc, sentence_spans: List[Entity], sentence_size=10) -> List[Entity]:
        total_sentences = len(sentence_spans)
        for i in range(0, total_sentences, sentence_size):
            sent_start = sentence_spans[i]
            sent_end = sentence_spans[min(i + sentence_size, total_sentences) - 1]
            start = sent_start.spans[0].start
            end = sent_end.spans[-1].end
            subset = doc.symbols[start:end]
            #file_name = doc.TitlesFieldName
            self.ner_instance.generate_txt(self.curr_file, subset, start, end)
            #self.ner_instance.run_matIE()
        return

# Ensure you have the NER class defined in the same file or imported correctly.
# Here you would place the NER class definition if it's not in a separate file.

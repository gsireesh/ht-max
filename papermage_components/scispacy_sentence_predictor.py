"""

Sentence Splitter using SciSpacy
@Yueheng Zhang
Adapted from PySBD sentence splitter by
@shannons, @kylel

"""

import itertools
from typing import List, Tuple

import numpy as np
import pysbd
import scispacy
import spacy
from collections import defaultdict
import os
import transformers
import subprocess

from papermage.magelib import (
    Document,
    Entity,
    PagesFieldName,
    Span,
    TokensFieldName,
    WordsFieldName,
)
from papermage.predictors import BasePredictor
from papermage.utils.merge import cluster_and_merge_neighbor_spans
import spacy


class SciSpacySentencePredictor(BasePredictor):
    """Sentence Boundary based on scispacy

    Examples:
        >>> doc: Document = parser.parse("path/to/pdf")
        >>> predictor = scispacySentenceBoundaryPredictor()
        >>> sentence_spans = predictor.predict(doc)
        >>> doc.annotate(sentences=sentence_spans)
    """

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [TokensFieldName]  # type: ignore

    def __init__(self, model_name = "en_core_sci_scibert", outDIR = "") -> None:
        scispacy = spacy.load(model_name)
        self.model = scispacy
        self.outDIR = outDIR
        #self._segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)

    def split_token_based_on_sentences_boundary(self, words: List[str]) -> List[Tuple[int, int]]:
        """
        Split a list of words into a list of (start, end) indices, indicating
        the start and end of each sentence.
        Duplicate of https://github.com/allenai/VILA/\blob/dd242d2fcbc5fdcf05013174acadb2dc896a28c3/src/vila/dataset/preprocessors/layout_indicator.py#L14      # noqa: E501

        Returns: List[Tuple(int, int)]
            a list of (start, end) for token indices within each sentence
        """

        if len(words) == 0:
            return [(0, 0)]
        combined_words = " ".join(words)
        self.combined_words = combined_words

        char2token_mask = np.zeros(len(combined_words), dtype=np.int64)

        acc_word_len = 0
        for idx, word in enumerate(words):
            word_len = len(word) + 1
            char2token_mask[acc_word_len : acc_word_len + word_len] = idx
            acc_word_len += word_len

        #segmented_sentences = self._segmenter.segment(combined_words)
############ use scispacy models start ###############

        #spacy_model1 = spacy.load("en_core_sci_scibert")
        doc1 = self.model(combined_words)

        segmented_sentences = list(doc1.sents)
        sent_boundary = [(ele.start_char, ele.end_char) for ele in segmented_sentences]
        split = []
        token_id_start = 0
        for start, end in sent_boundary:
            token_id_end = char2token_mask[start:end].max()
            if end + 1 >= len(char2token_mask) or char2token_mask[end + 1] != token_id_end:
                token_id_end += 1  # (Including the end)
            split.append((token_id_start, token_id_end))
            token_id_start = token_id_end
        return split
        # spacy_model2 = spacy.load("en_core_sci_md")
        # doc2 = spacy_model2(combined_words)

        # spacy_model3 = spacy.load("en_core_sci_lg")
        # doc3 = spacy_model3(combined_words)        
        
############ use scispacy models end ###############
        # sent_boundary = [(ele.start, ele.end) for ele in segmented_sentences]

        # split = []
        # token_id_start = 0
        # for start, end in sent_boundary:
        #     token_id_end = char2token_mask[start:end].max()
        #     if end + 1 >= len(char2token_mask) or char2token_mask[end + 1] != token_id_end:
        #         token_id_end += 1  # (Including the end)
        #     split.append((token_id_start, token_id_end))
        #     token_id_start = token_id_end
        # return split

    # split the Document and run zhisong's model
    # return the entities spans
    def _predict(self, doc: Document) -> List[Entity]:
        if hasattr(doc, WordsFieldName):
            words = [word.text for word in getattr(doc, WordsFieldName)]
            attr_name = WordsFieldName
            # `words` is preferred as it should has better reading
            # orders and text representation
        else:
            words = [token.text for token in doc.tokens]
            attr_name = TokensFieldName

        split = self.split_token_based_on_sentences_boundary(words)

        # split the sentences into 100 groups 
        
        sentence_spans: List[Entity] = []
        for start, end in split:
            if end - start == 0:
                continue
            if end - start < 0:
                raise ValueError

            cur_spans = getattr(doc, attr_name)[start:end]

            all_token_spans = list(itertools.chain.from_iterable([ele.spans for ele in cur_spans]))
            results = cluster_and_merge_neighbor_spans(all_token_spans)
            sentence_spans.append(Entity(spans=results.merged))

            # for span in sentence_spans from zhisong's model:
                # entity_predict = List[Entity] 
                # entity_predict.append(Entity(spans= zhisong's spans list))
                    #eg.[33565,33577]
                    #eg.[33581,33595]
                # sentence_spans.append(Entity(spans=results.merged))
        #print('_predict, output_dir',self.outDIR)
        self.entity_predict(doc, sentence_spans, 10)
        return sentence_spans

############################## Entity prediction ####################################
    # split the Document into subsets and run zhisong's model
    # return the entities spans
    def entity_predict(self, doc, sentence_spans: List[Entity], sentence_size = 10) -> List[Entity]:
        total_sentences = len(sentence_spans)
        sentences_span_idx = defaultdict(int) # start index of sentence span        print("total_sentences",total_sentences)
        for i in range(0, total_sentences, sentence_size):
            sent_start = sentence_spans[i]
            sent_end = sentence_spans[i + sentence_size] if i + sentence_size <= total_sentences - 1 else sentence_spans[total_sentences - 1]
            start = sent_start.spans[0].start
            # find the end index
            end = sent_end.spans[0].end
            sentences_span_idx[i] = start

            subset = doc.symbols[start: end]
            # generate .txt files from a subset 
            self.generate_txt(subset,start,end)
            # run matIE for the subset
            #self.run_matIE(model_dir, vocab_dir, input_folder, output_folder, gpu_id, decode_script)

            # process results from matIE and generate entity list 
            #self.process_MatIE()
            #entity_spans = self.process_MatIE() to be implemented
        # return entity_spans
        return
    
    # def generate_txt(self, subset, start, end):
    #     # Extract the sentences list from the data and Replace newline characters with spaces
    #     sentences = subset.replace("\n", " ")

    #     # Specify the path of the file where you want to save the sentences
    #     #file_path = f'data/AM_Creep_Papers_parsed_Feb25/{start}-{end}.txt'
    #     file_path = f'{self.DIR_name}/{start}-{end}.txt'
    #     # Write the sentences to the specified file
    #     with open(file_path, 'w', encoding='utf-8') as file:
    #         file.write(sentences)

    #     print(f"Sentences have been successfully written to {file_path}")
    def generate_txt(self, subset, start, end):
        # Extract the sentences list from the data and Replace newline characters with spaces
        sentences = subset.replace("\n", " ")

        # Specify the path of the file where you want to save the sentences
        file_path = f'{self.outDIR}/{start}-{end}.txt'
        
        # Ensure the directory exists
        os.makedirs(self.outDIR, exist_ok=True)
        # print('generate_txt in self.outDIR',self.outDIR)
        # Write the sentences to the specified file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(sentences)

        print(f"Sentences have been successfully written to {file_path}")


    def run_matIE(self, model_dir, vocab_dir, input_folder, output_folder, gpu_id, decode_script):
        # model_dir = '/content/drive/MyDrive/src/model'
        # vocab_dir = '/content/drive/MyDrive/src/vpack_mat'
        # gpu_id = '0'
        # decode_script = '/content/drive/MyDrive/src/decode.sh'
        # DIR_NAME = "/content/drive/MyDrive/src/data230707/NER_Results_2"

        model_dir = model_dir
        vocab_dir = vocab_dir
        gpu_id = gpu_id
        decode_script = decode_script
        DIR_NAME = DIR_NAME

        for dir_name in os.listdir(DIR_NAME):
            process_dir_name = dir_name.replace(" ", "_")
            
            # Rename the folder
            original_dir_path = os.path.join(DIR_NAME, dir_name)
            processed_dir_path = os.path.join(DIR_NAME, process_dir_name)
            os.rename(original_dir_path, processed_dir_path)

            input_folder = processed_dir_path
            output_folder = processed_dir_path

            self.process_files_multiprocess(model_dir, vocab_dir, input_folder, output_folder, gpu_id, decode_script)

    def process_files_multiprocess(self, model_dir, vocab_dir, input_folder, output_folder, gpu_id, decode_script):
        # Set environment variables
        env_vars = os.environ.copy()
        env_vars['MODEL_DIR'] = model_dir
        env_vars['VOCAB_DIR'] = vocab_dir
        env_vars['INPUT_DIR'] = input_folder
        env_vars['OUTPUT_DIR'] = output_folder
        env_vars['CUDA_VISIBLE_DEVICES'] = gpu_id

        # Construct and run the command to generate input.json
        cmd_generate_json = f"python3 -m mspx.tools.al.utils_brat cmd:b2z input_path:{input_folder}/ output_path:{output_folder}/input.json delete_nils:1 convert.toker:nltk"
        
        # Execute the command
        subprocess.run(cmd_generate_json, shell=True, env=env_vars)

        # Make sure decode.sh is executable
        subprocess.run(['chmod', '+x', decode_script], check=True)

        # Run the decode script
        subprocess.run(decode_script, shell=True, env=env_vars)
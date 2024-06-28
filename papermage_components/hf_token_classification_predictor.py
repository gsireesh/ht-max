from dataclasses import dataclass
import re
from typing import List

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from papermage.magelib import Document, Entity, Metadata, Span
from papermage.predictors import BasePredictor


SENTENCE_BATCH_SIZE = 4


@dataclass
class EntityCharSpan:
    e_type: str
    start_char: int
    end_char: int


def get_char_spans_from_labels(
    label_list: list[str], offset_mapping: list[list[int]], skip_labels=("O",)
) -> list[EntityCharSpan]:
    annotations_list = []
    current_annotation = None

    for label, (offset_start, offset_end) in zip(label_list, offset_mapping):
        cleaned_label = re.sub("[BIO]-", "", label)
        if current_annotation is None:
            current_annotation = EntityCharSpan(
                e_type=cleaned_label, start_char=offset_start, end_char=offset_end
            )
            continue
        elif cleaned_label != current_annotation.e_type:
            if current_annotation.e_type not in skip_labels:
                annotations_list.append(current_annotation)
            current_annotation = EntityCharSpan(
                e_type=cleaned_label, start_char=offset_start, end_char=offset_end
            )
        elif cleaned_label == current_annotation.e_type:
            current_annotation.end_char = offset_end
        else:
            raise AssertionError("Unexpected case!!")
    return annotations_list


def map_entities_to_sentence_spans(
    sentence: Entity, entities: list[EntityCharSpan]
) -> list[Entity]:
    all_entities = []

    # compute a map of offsets from the beginning of the sentence to every position in it
    sentence_spans = sentence.spans
    assert len(sentence.text) == sum([span.end - span.start for span in sentence_spans])
    offset_to_span_map = {}
    sentence_offset = 0
    for span_index, span in enumerate(sentence_spans):
        for span_offset in range(span.start, span.end + 1):
            offset_to_span_map[sentence_offset] = (span_index, span_offset)
            sentence_offset += 1

    # using the offset map, get a list of spans for each entity.
    for entity in entities:
        start_span_index, start_span_offset = offset_to_span_map[entity.start_char]
        entity_start = start_span_offset
        end_span_index, end_span_offset = offset_to_span_map[entity.end_char]
        entity_end = end_span_offset

        if start_span_index != end_span_index:
            start_span = Span(entity_start, sentence_spans[start_span_index.end])
            end_span = Span(sentence_spans[end_span_index.start], entity_end)
            intervening_spans = [
                Span(sentence_spans[i].start, sentence_spans[i].end)
                for i in range(start_span_index + 1, end_span_index)
            ]
            spans = [start_span] + intervening_spans + [end_span]
        else:
            spans = [Span(entity_start, entity_end)]

        all_entities.append(Entity(spans=spans, metadata=Metadata(entity_type=entity.e_type)))
    return all_entities


class HfTokenClassificationPredictor(BasePredictor):
    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return ["reading_order_sections"]

    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        self.id2label = self.model.config.id2label

    def tag_entities(self, sentences: list[str]) -> list[list[EntityCharSpan]]:
        tokenized = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )
        offset_mapping = tokenized.offset_mapping.tolist()
        model_output = self.model(
            input_ids=tokenized.input_ids.to(self.device),
            attention_mask=tokenized.attention_mask.to(self.device),
        )
        label_idxs = torch.argmax(model_output.logits, dim=-1).tolist()
        label_lists = [
            [
                self.id2label[idx]
                for idx, attention_value in zip(label_list, attention_mask)
                if attention_value == 1
            ]
            for (label_list, attention_mask) in zip(label_idxs, tokenized.attention_mask)
        ]
        entity_char_spans = [
            get_char_spans_from_labels(label_list, instance_offset_mapping)
            for (label_list, instance_offset_mapping) in zip(label_lists, offset_mapping)
        ]
        return entity_char_spans

    def _predict(self, doc: Document) -> list[Entity]:

        all_entities = []

        # some sentences intersect with multiple paragraphs, and we don't want to process them twice
        already_processed_sentences = set()
        for para_idx, paragraph in tqdm(enumerate(doc.reading_order_sections)):

            paragraph_sentences = [
                sentence
                for sentence in paragraph.sentences
                if sentence not in already_processed_sentences
            ]
            if not paragraph_sentences:
                continue

            sentence_texts = [sentence.text.replace("\n", " ") for sentence in paragraph_sentences]

            entities_by_sentence = self.tag_entities(sentence_texts)
            for sentence, sentence_entities in zip(paragraph_sentences, entities_by_sentence):
                all_entities.extend(map_entities_to_sentence_spans(sentence, sentence_entities))
            already_processed_sentences.update(paragraph_sentences)

        return all_entities

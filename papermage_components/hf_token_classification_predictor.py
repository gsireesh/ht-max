import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from papermage_components.interfaces.token_classification_predictor import (
    EntityCharSpan,
    TokenClassificationPredictorABC,
)


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


class HfTokenClassificationPredictor(TokenClassificationPredictorABC):
    def __init__(self, model_name, device):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        self.id2label = self.model.config.id2label

    @property
    def predictor_identifier(self) -> str:
        return self.model_name

    def tag_entities_in_batch(self, batch: list[str]) -> list[list[EntityCharSpan]]:
        tokenized = self.tokenizer(
            batch,
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

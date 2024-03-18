import itertools
from typing import Optional

from ncls import NCLS
import numpy as np

from papermage import Document, Box, Entity, Span
from papermage.utils.merge import cluster_and_merge_neighbor_spans

from papermage_components.constants import MAT_IE_TYPES


def get_spans_from_boxes(doc: Document, boxes: list[Box]):
    intersecting_tokens = doc.intersect_by_box(query=Entity(boxes=boxes), name="tokens")
    token_spans = list(itertools.chain(*(token.spans for token in intersecting_tokens)))
    clustered_token_spans = cluster_and_merge_neighbor_spans(token_spans).merged
    filtered_for_strays = [
        merged for merged in clustered_token_spans if merged.end - merged.start > 1
    ]
    return filtered_for_strays


def get_span_by_box(box, doc) -> Optional[Span]:
    overlapping_tokens = doc.intersect_by_box(Entity(boxes=[box]), "tokens")
    token_spans = []
    for token in overlapping_tokens:
        token_spans.extend(token.spans)
    if token_spans:
        return Span.create_enclosing_span(token_spans)
    else:
        return None


def get_text_in_box(box, doc):
    cell_span = get_span_by_box(box, doc)
    return doc.symbols[cell_span.start : cell_span.end] if cell_span is not None else ""


def globalize_bbox_coordinates(bbox, context_box, doc):
    page_width, page_height = doc.pages[context_box.page].images[0].pilimage.size
    bbox_left = context_box.l + (bbox[0] / page_width)
    bbox_top = context_box.t + (bbox[1] / page_height)
    bbox_width = (bbox[2] - bbox[0]) / page_width
    bbox_height = (bbox[3] - bbox[1]) / page_height
    return Box(bbox_left, bbox_top, bbox_width, bbox_height, page=context_box.page)


def get_text_from_localized_bbox(bbox, context_box, doc):
    global_box = globalize_bbox_coordinates(bbox, context_box, doc)
    return get_text_in_box(global_box, doc)


def merge_overlapping_entities(entities):
    starts = []
    ends = []
    ids = []

    for id, entity in enumerate(entities):
        for span in entity.spans:
            starts.append(span.start)
            ends.append(span.end)
            ids.append(id)

    index = NCLS(
        np.array(starts, dtype=np.int32),
        np.array(ends, dtype=np.int32),
        np.array(ids, dtype=np.int32),
    )

    merged_entities = []
    consumed_entities = set()
    for entity in entities:
        if entity in consumed_entities:
            continue
        for span in entity.spans:
            match_ids = [
                matched_id for _start, _end, matched_id in index.find_overlap(span.start, span.end)
            ]
            overlapping_entities = [entities[i] for i in match_ids]
            if len(overlapping_entities) == 1:
                merged_entities.append(entity)
            elif len(overlapping_entities) > 1:
                all_spans = list(
                    itertools.chain(*[entity.spans for entity in overlapping_entities])
                )
                if entity.boxes is not None:
                    all_boxes = list(
                        itertools.chain(*[entity.boxes for entity in overlapping_entities])
                    )
                else:
                    all_boxes = None

                merged_entities.append(
                    Entity(
                        spans=[Span.create_enclosing_span(all_spans)],
                        boxes=all_boxes,
                        metadata=overlapping_entities[0].metadata,
                    )
                )
                consumed_entities.update(overlapping_entities)
                break

    return merged_entities


def annotate_entities_on_doc(entities_by_type, spacy_doc, para_offset):
    all_spans = []
    for e_type, entities in entities_by_type.items():
        if not entities:
            continue
        for entity in entities:
            e_start_char = entity.spans[0].start - para_offset
            e_end_char = entity.spans[0].end - para_offset

            span = spacy_doc.char_span(e_start_char, e_end_char, label=e_type)
            if span is not None:
                all_spans.append(span)
    spacy_doc.set_ents(all_spans)


def visualize_highlights(paragraph_entity, spacy_pipeline):
    entities_by_type = {}
    for e in paragraph_entity.annotation_highlights:
        e_type = e.metadata["annotation_type"]
        if e_type not in entities_by_type:
            entities_by_type[e_type] = []
        entities_by_type[e_type].append(e)

    para_doc = spacy_pipeline(paragraph_entity.text.replace("\n", " "))
    para_offset = paragraph_entity.spans[0].start
    if entities_by_type:
        annotate_entities_on_doc(entities_by_type, para_doc, para_offset)
    return para_doc


def visualize_matIE_annotations(paragraph_entity, spacy_pipeline):
    entities_by_type = {e_type: getattr(paragraph_entity, e_type) for e_type in MAT_IE_TYPES}
    para_doc = spacy_pipeline(paragraph_entity.text.replace("\n", " "))
    para_offset = paragraph_entity.spans[0].start
    annotate_entities_on_doc(entities_by_type, para_doc, para_offset)
    return para_doc


def get_table_image(table_entity: Entity, doc: Document, page_image=None):
    if len(table_entity.boxes) > 1:
        raise AssertionError("Table has more than one box!!")
    box = table_entity.boxes[0]
    if page_image is None:
        page_image = doc.pages[box.page].images[0].pilimage
    page_w, page_h = page_image.size
    table_image = page_image.crop(
        (
            box.l * page_w,
            box.t * page_h,
            (box.l + box.w) * page_w,
            (box.t + box.h) * page_h,
        )
    )
    return table_image

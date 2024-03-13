import itertools

from ncls import NCLS
import numpy as np

from papermage import Entity, Span
from papermage.utils.merge import cluster_and_merge_neighbor_spans


def get_spans_from_boxes(doc, boxes):
    intersecting_tokens = doc.intersect_by_box(query=Entity(boxes=boxes), name="tokens")
    token_spans = list(itertools.chain(*(token.spans for token in intersecting_tokens)))
    clustered_token_spans = cluster_and_merge_neighbor_spans(token_spans).merged
    filtered_for_strays = [
        merged for merged in clustered_token_spans if merged.end - merged.start > 1
    ]
    return filtered_for_strays


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

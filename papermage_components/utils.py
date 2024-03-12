import itertools

from papermage import Entity
from papermage.utils.merge import cluster_and_merge_neighbor_spans


def get_spans_from_boxes(doc, boxes):
    intersecting_tokens = doc.intersect_by_box(query=Entity(boxes=boxes), name="tokens")
    token_spans = list(itertools.chain(*(token.spans for token in intersecting_tokens)))
    clustered_token_spans = cluster_and_merge_neighbor_spans(token_spans).merged
    filtered_for_strays = [
        merged for merged in clustered_token_spans if merged.end - merged.start > 1
    ]
    return filtered_for_strays

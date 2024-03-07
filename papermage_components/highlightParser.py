import itertools
import os

import fitz
from papermage.magelib import (
    Box,
    Document,
    EntitiesFieldName,
    Entity,
    Metadata,
    Span,
)
from papermage.parsers.parser import Parser
from papermage.utils.merge import cluster_and_merge_neighbor_spans

HighlightsFieldName = "annotation_highlights"

FITZ_HIGHLIGHT_FIELD_NAME = "Highlights"
ANNOTATION_TYPE_KEY = "annotation_type"
B_VALUE_TO_TYPE = {
    1.0: "structure",
    0.15685999393463135: "property",
    0.0: "characterization",
    0.007843020372092724: "processing",
    0.2156829982995987: "materials",
    0.5254970192909241: "info",
}


def convert_rect_to_papermage(rect, page, page_number):
    left = rect[0] / page.rect.width
    top = rect[1] / page.rect.width
    width = (rect[2] - rect[0]) / page.rect.width
    height = (rect[3] - rect[1]) / page.rect.height

    return Box(l=left, t=top, w=width, h=height, page=page_number)


def get_highlight_spans(boxes, doc):
    intersecting_tokens = doc.intersect_by_box(query=Entity(boxes=boxes), name="tokens")
    token_spans = itertools.chain((token.spans for token in intersecting_tokens))
    clustered_token_spans = cluster_and_merge_neighbor_spans(token_spans)
    return clustered_token_spans.merged


class FitzHighlightParser(Parser):
    def __init__(self, annotated_pdf_directory: str):
        self.annotated_pdf_directory = annotated_pdf_directory

    def parse(self, input_pdf_path: str, doc: Document) -> Document:
        pdf_filename = os.path.basename(input_pdf_path)
        annotated_filename = os.path.join(self.annotated_pdf_directory, f"annotated_{pdf_filename}")

        with fitz.open(annotated_filename) as pdf:
            for page_number, page in enumerate(pdf):
                for annotation in page.annots():
                    if annotation.type[1] != FITZ_HIGHLIGHT_FIELD_NAME:
                        continue

                    # get annotation boxes
                    entity_boxes = []
                    vertices = annotation.vertices

                    assert len(vertices) % 4 == 0

                    if len(vertices) == 4:
                        box = fitz.Quad(vertices).rect
                        entity_boxes.append(convert_rect_to_papermage(box, page, page_number))
                    else:
                        for j in range(0, len(vertices), 4):
                            box = fitz.Quad(vertices[j, j + 4]).rect
                            entity_boxes.append(convert_rect_to_papermage(box, page, page_number))

                    # get annotation color, and then type
                    color = annotation.colors["stroke"]
                    annotation_type = B_VALUE_TO_TYPE[color]

                    entity_metadata = Metadata(
                        **{"annotation_color": color, ANNOTATION_TYPE_KEY: annotation_type}
                    )

                    entity_spans = get_highlight_spans(entity_boxes, doc)

                    highlight_entity = Entity(
                        spans=entity_spans,
                        boxes=entity_boxes,
                        images=None,
                        metadata=entity_metadata,
                    )

        return doc

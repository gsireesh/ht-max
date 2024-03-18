import os

import fitz
from papermage.magelib import (
    Box,
    Document,
    Entity,
    Metadata,
)
from papermage.parsers.parser import Parser
from papermage_components.utils import get_spans_from_boxes

HighlightsFieldName = "annotation_highlights"

FITZ_HIGHLIGHT_FIELD_NAME = "Highlight"
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
    top = rect[1] / page.rect.height
    width = (rect[2] - rect[0]) / page.rect.width
    height = (rect[3] - rect[1]) / page.rect.height

    return Box(l=left, t=top, w=width, h=height, page=page_number)


def vertical_shrink(box, factor):
    top_diff = (1 - factor) * box.h / 2
    return Box(box.l, box.t + top_diff, box.w, box.h * factor, box.page)


def get_highlight_entities_from_pdf(pdf_filename: str, doc: Document) -> list[Entity]:
    highlight_entities = []
    with fitz.open(pdf_filename) as pdf:
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
                    entity_boxes.append(
                        vertical_shrink(convert_rect_to_papermage(box, page, page_number), 0.5)
                    )
                else:
                    for j in range(0, len(vertices), 4):
                        box = fitz.Quad(vertices[j : j + 4]).rect
                        entity_boxes.append(
                            vertical_shrink(convert_rect_to_papermage(box, page, page_number), 0.5)
                        )

                # get annotation color, and then type
                color = annotation.colors["stroke"]
                annotation_type = B_VALUE_TO_TYPE[color[2]]

                entity_spans = get_spans_from_boxes(doc, entity_boxes)

                entity_metadata = Metadata(
                    **{"annotation_color": color, ANNOTATION_TYPE_KEY: annotation_type}
                )

                highlight_entity = Entity(
                    spans=entity_spans,
                    boxes=entity_boxes,
                    images=None,
                    metadata=entity_metadata,
                )
                highlight_entities.append(highlight_entity)
    return highlight_entities


class FitzHighlightParser(Parser):
    def __init__(self, annotated_pdf_directory: str):
        self.annotated_pdf_directory = annotated_pdf_directory

    def parse(self, input_pdf_path: str, doc: Document) -> Document:
        pdf_filename = os.path.basename(input_pdf_path)
        annotated_filename = os.path.join(self.annotated_pdf_directory, f"annotated_{pdf_filename}")

        if not os.path.exists(annotated_filename):
            print("No annotated file found, skipping...")
            return doc

        highlight_entities = get_highlight_entities_from_pdf(annotated_filename, doc)

        doc.annotate_layer(HighlightsFieldName, highlight_entities, require_disjoint=False)

        return doc

"""
@gsireesh
"""

from collections import defaultdict
import itertools
import json
import os
from tempfile import NamedTemporaryFile
from typing import Any, Optional
import xml.etree.ElementTree as ET

from grobid_client.grobid_client import GrobidClient
from papermage.magelib import (
    Box,
    Document,
    Entity,
    Metadata,
)
from papermage.parsers.parser import Parser

from papermage_components.utils import get_spans_from_boxes, merge_overlapping_entities


NS = {"tei": "http://www.tei-c.org/ns/1.0"}


IN_PARAGRAPH_DISTANCE_TOLERANCE = 0.025


def get_page_dimensions(root: ET.Element) -> dict[int, tuple[float, float]]:
    page_size_root = root.find(".//tei:facsimile", NS)
    assert page_size_root is not None, "No facsimile found in Grobid XML"

    page_size_data = page_size_root.findall(".//tei:surface", NS)
    page_sizes = dict()
    for data in page_size_data:
        page_sizes[int(data.attrib["n"]) - 1] = (
            float(data.attrib["lrx"]),
            float(data.attrib["lry"]),
        )

    return page_sizes


def parse_grobid_coords(
    coords_string: str, page_sizes: dict[int, tuple[float, float]]
) -> list[Box]:
    boxes = []
    for box_coords in coords_string.split(";"):
        coords_list = box_coords.split(",")
        page_number = int(coords_list[0]) - 1
        page_width, page_height = page_sizes[page_number]

        l = float(coords_list[1]) / page_width
        t = float(coords_list[2]) / page_height
        w = float(coords_list[3]) / page_width
        h = float(coords_list[4]) / page_height

        boxes.append(Box(l, t, w, h, page_number))
    return boxes


def get_all_child_sentence_boxes(element, page_dimensions):
    sentences = element.findall(".//tei:s[@coords]", NS)
    element_coords = [e.attrib["coords"] for e in sentences]
    element_boxes = list(
        itertools.chain(
            *[parse_grobid_coords(coord_string, page_dimensions) for coord_string in element_coords]
        )
    )
    return element_boxes


def get_abstract_box(root: ET.Element, page_dimensions: dict[int, tuple[float, float]]) -> Box:
    abstract_tags = root.findall(".//tei:teiHeader/tei:profileDesc/tei:abstract", NS)
    all_abstract_boxes = []

    for abstract_tag in abstract_tags:
        all_abstract_boxes.extend(get_all_child_sentence_boxes(abstract_tag, page_dimensions))

    abstract_box = Box.create_enclosing_box(all_abstract_boxes)
    return abstract_box


def get_coords_by_section(
    root: ET.Element, page_dimensions: dict[int, tuple[float, float]]
) -> dict[str, list[list[Box]]]:
    section_divs = root.findall(".//tei:text/tei:body/tei:div", NS)
    coords_by_section = {}
    for div in section_divs:
        title_element = div.find("./tei:head", NS)
        if title_element is not None:
            title_text = title_element.text
            title_coords = title_element.attrib["coords"]
            all_coords = [parse_grobid_coords(title_coords, page_dimensions)]
        else:
            title_text = "Unknown Section"
            all_coords = []

        section_paragraphs = div.findall("./tei:p", NS)
        for paragraph in section_paragraphs:
            paragraph_boxes = get_all_child_sentence_boxes(paragraph, page_dimensions)
            all_coords.append(paragraph_boxes)

        coords_by_section[title_text] = all_coords
    return coords_by_section


def box_span_intersects(span1, span2, tol=0.0):
    start1, end1 = span1
    start2, end2 = span2
    return (start1 - tol <= start2 <= end1 + tol) or (start2 - tol <= start1 <= end2 + tol)


def update_cover_span(cover, span):
    new_cover_span = (min(cover[0], span[0]), max(cover[1], span[1]))
    return new_cover_span


def group_boxes_by_column(boxes: list[Box]):
    horizontal_covers = []
    boxes_by_group = defaultdict(list)

    for box in boxes:
        left_limit = box.l
        right_limit = box.l + box.w
        box_span = (left_limit, right_limit)

        if horizontal_covers:
            for i, cover in enumerate(horizontal_covers):
                if (
                    box_span_intersects(cover, box_span, tol=IN_PARAGRAPH_DISTANCE_TOLERANCE)
                    and box.t - boxes_by_group[i][-1].t > -IN_PARAGRAPH_DISTANCE_TOLERANCE
                ):
                    horizontal_covers[i] = update_cover_span(cover, box_span)
                    # this break implicitly *assumes* a columnar structure - if we e.g. have a piece
                    # of text that spans two columns, we won't find it
                    boxes_by_group[i].append(box)
                    break
            else:
                boxes_by_group[len(horizontal_covers)].append(box)
                horizontal_covers.append(box_span)
        else:
            boxes_by_group[len(horizontal_covers)].append(box)
            horizontal_covers.append((left_limit, right_limit))

    return [Box.create_enclosing_box(box_group) for box_group in boxes_by_group.values()]


def segment_and_consolidate_boxes(
    section_boxes: list[list[Box]], section_name: str
) -> list[list[Box]]:
    consolidated_boxes = []
    for paragraph_boxes in section_boxes:
        boxes_by_page = defaultdict(list)
        for box in paragraph_boxes:
            boxes_by_page[box.page].append(box)

        for _, page_boxes in boxes_by_page.items():
            grouped_boxes = group_boxes_by_column(page_boxes)
            consolidated_boxes.append(grouped_boxes)

    return consolidated_boxes


class GrobidReadingOrderParser(Parser):
    def __init__(
        self,
        grobid_server_url,
        check_server: bool = True,
        xml_out_dir: Optional[str] = None,
        **grobid_config: Any
    ):
        self.grobid_config = {
            "grobid_server": grobid_server_url,
            "batch_size": 1000,
            "sleep_time": 5,
            "timeout": 6000,
            "coordinates": sorted({"head", "p", "s", "ref", "body", "item", "persName"}),
            **grobid_config,
        }
        assert "coordinates" in self.grobid_config, "Grobid config must contain 'coordinates' key"

        with NamedTemporaryFile(mode="w", delete=False) as f:
            json.dump(self.grobid_config, f)
            config_path = f.name

        self.client = GrobidClient(config_path=config_path, check_server=check_server)

        self.xml_out_dir = xml_out_dir
        os.remove(config_path)

    def parse(self, input_pdf_path: str, doc: Document) -> Document:
        assert doc.symbols != ""

        (_, _, xml) = self.client.process_pdf(
            service="processFulltextDocument",
            pdf_file=input_pdf_path,
            generateIDs=False,
            consolidate_header=False,
            consolidate_citations=False,
            include_raw_citations=False,
            include_raw_affiliations=False,
            tei_coordinates=True,
            segment_sentences=True,
        )
        assert xml is not None, "Grobid returned no XML"

        if self.xml_out_dir:
            os.makedirs(self.xml_out_dir, exist_ok=True)
            xml_file = os.path.join(
                self.xml_out_dir, os.path.basename(input_pdf_path).replace(".pdf", ".xml")
            )
            with open(xml_file, "w") as f_out:
                f_out.write(xml)

        xml_root = ET.fromstring(xml)
        page_dimensions = get_page_dimensions(xml_root)
        section_to_boxes = get_coords_by_section(xml_root, page_dimensions)

        consolidated_boxes = {
            section: segment_and_consolidate_boxes(section_boxes, section)
            for section, section_boxes in section_to_boxes.items()
        }

        abstract_box = get_abstract_box(xml_root, page_dimensions)
        consolidated_boxes["abstract"] = [[abstract_box]]

        paragraph_entities = []
        for section_number, (section, section_paragraph_boxes) in enumerate(
            consolidated_boxes.items()
        ):
            for paragraph_order, paragraph_boxes in enumerate(section_paragraph_boxes):
                paragraph_spans = get_spans_from_boxes(doc, paragraph_boxes)
                paragraph_entity = Entity(
                    boxes=paragraph_boxes,
                    spans=paragraph_spans,
                    metadata=Metadata(
                        section_name=section,
                        section_reading_order=section_number,
                        paragraph_reading_order=paragraph_order,
                    ),
                )
                paragraph_entities.append(paragraph_entity)

        merged_paragraphs = merge_overlapping_entities(paragraph_entities)
        doc.annotate_layer("reading_order_sections", merged_paragraphs)

        return doc

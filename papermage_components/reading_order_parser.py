"""
@gsireesh
"""

from collections import defaultdict
import itertools
import json
import os
from tempfile import NamedTemporaryFile
from typing import Any, Optional
import xml.etree.ElementTree as et

from grobid_client.grobid_client import GrobidClient

from papermage.magelib import (
    Document,
    Entity,
    Metadata,
)
from papermage.magelib.box import Box
from papermage.parsers.grobid_parser import GROBID_VILA_MAP
from papermage.parsers.parser import Parser


NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def get_page_dimensions(root: et.Element) -> dict[int, tuple[float, float]]:
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


def get_coords_by_section(
    root: et.Element, page_dimensions: dict[int, tuple[float, float]]
) -> dict[str, list[Box]]:
    section_divs = root.findall(".//tei:text/tei:body/tei:div", NS)
    coords_by_section = {}
    for div in section_divs:
        title_element = div.find("./tei:head", NS)
        title_text = title_element.text
        title_coords = title_element.attrib["coords"]
        sentence_elements = div.findall(".//tei:s[@coords]", NS)
        all_coords = [title_coords] + [e.attrib["coords"] for e in sentence_elements]
        section_boxes = list(
            itertools.chain(
                *[parse_grobid_coords(coord_string, page_dimensions) for coord_string in all_coords]
            )
        )

        coords_by_section[title_text] = section_boxes
    return coords_by_section


def intersects(span1, span2, tol=0.0):
    start1, end1 = span1
    start2, end2 = span2
    return (start1 - tol <= start2 <= end1 + tol) or (start2 - tol <= start1 <= end2 + tol)


def update_cover_span(cover, span):
    new_cover_span = (min(cover[0], span[0]), max(cover[1], span[1]))
    return new_cover_span


def group_boxes_by_column(boxes: list[Box]):
    horizontal_covers = []
    groups_list = []
    for box in boxes:
        left_limit = box.l
        right_limit = box.l + box.w
        box_span = (left_limit, right_limit)

        if horizontal_covers:
            for i, cover in enumerate(horizontal_covers):
                if intersects(cover, box_span, tol=0.01):
                    horizontal_covers[i] = update_cover_span(cover, box_span)
                    # this break implicitly *assumes* a columnar structure - if we e.g. have a piece
                    # of text that spans two columns, we won't find it
                    groups_list.append(i)
                    break
            else:
                groups_list.append(len(horizontal_covers))
                horizontal_covers.append(box_span)
        else:
            groups_list.append(len(horizontal_covers))
            horizontal_covers.append((left_limit, right_limit))

    boxes_by_group = defaultdict(list)
    for group, box in zip(groups_list, boxes):
        boxes_by_group[group].append(box)

    return [Box.create_enclosing_box(box_group) for box_group in boxes_by_group.values()]


def segment_and_consolidate_boxes(section_boxes: list[Box], section_name: str) -> list[Box]:
    boxes_by_page = defaultdict(list)
    for box in section_boxes:
        boxes_by_page[box.page].append(box)

    consolidated_boxes = []
    for _, page_boxes in boxes_by_page.items():
        grouped_boxes = group_boxes_by_column(page_boxes)
        consolidated_boxes.extend(grouped_boxes)

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
            "timeout": 60,
            "coordinates": sorted(
                set((*GROBID_VILA_MAP.keys(), "s", "ref", "body", "item", "persName"))
            ),
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

        xml_root = et.fromstring(xml)
        page_dimensions = get_page_dimensions(xml_root)
        section_to_boxes = get_coords_by_section(xml_root, page_dimensions)

        consolidated_boxes = {
            section: segment_and_consolidate_boxes(section_boxes, section)
            for section, section_boxes in section_to_boxes.items()
        }

        section_entities = [
            Entity(boxes=boxes, metadata=Metadata(section_name=section_name, order=i))
            for i, (section_name, boxes) in enumerate(consolidated_boxes.items())
        ]

        doc.annotate_layer("reading_order_sections", section_entities)

        return doc

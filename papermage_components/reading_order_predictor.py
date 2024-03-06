"""
@gsireesh
"""


import json
import os
import re
import warnings
import xml.etree.ElementTree as et
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from grobid_client.grobid_client import GrobidClient


from papermage.magelib import (
    BlocksFieldName,
    Document,
    Entity,
    Metadata,
    PagesFieldName,
    RowsFieldName,
    Span,
    TokensFieldName,
)
from papermage.magelib.box import Box
from papermage.parsers.grobid_parser import GROBID_VILA_MAP
from papermage.predictors import BasePredictor
from papermage.utils.merge import cluster_and_merge_neighbor_spans

REQUIRED_DOCUMENT_FIELDS = [BlocksFieldName, PagesFieldName, RowsFieldName, TokensFieldName]

class GrobidReadingOrderPredictor(BasePredictor):

    def __init__(self, grobid_server_url, check_server: bool = True, xml_out_dir: Optional[str] =
    None, **grobid_config: Any):
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


    def _predict(self, input_pdf_path: str, doc: Document) -> List[Entity]:
        assert doc.symbols != ""
        for field in REQUIRED_DOCUMENT_FIELDS:
            assert field in doc.layers

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
            xmlfile = os.path.join(
                self.xml_out_dir, os.path.basename(input_pdf_path).replace(".pdf", ".xml")
            )
            with open(xmlfile, "w") as f_out:
                f_out.write(xml)

        print("wait here!")
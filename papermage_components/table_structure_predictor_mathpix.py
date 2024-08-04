from typing import List
import io
import torch
from torchvision import transforms
from transformers import TableTransformerForObjectDetection

from papermage import Box, Document, Entity, TablesFieldName
from papermage.predictors import BasePredictor
from papermage_components.interfaces.image_predictor import ImagePredictionResult, ImagePredictorABC
from papermage_components.utils import get_table_image, get_text_in_box, globalize_bbox_coordinates

import requests
import base64
import json
import csv
from collections import defaultdict
from PIL import Image


MATHPIX_ENDPOINT = "https://api.mathpix.com/v3/text"


def get_mathpix_input(encoded_image):
    json_data = {
        "src": f"data:image/jpeg;base64,{encoded_image}",
        "formats": ["text", "data"],
        "data_options": {
            "include_tsv": True,
        },
    }
    return json_data


def encode_image(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    image_bytes = buffer.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return encoded_image


def find_non_empty_indices(lst):
    return [index for index, value in enumerate(lst) if value != ""]


# Function to parse TSV data
def parse_caption_note(latex_input):
    parts = latex_input.split(r"\begin{tabular}")
    caption = parts[0].strip()
    post_table_content = parts[-1].split(r"\end{tabular}")[-1].strip()
    return caption, post_table_content


def parse_tsv(tsv_string):
    # Read the TSV data
    reader = csv.reader(tsv_string.splitlines(), delimiter="\t")

    # Extract headers
    headers = next(reader)

    # Extract rows
    rows = []
    for row in reader:
        if "" in row:
            idx_list = find_non_empty_indices(row)
            if "" == row[0]:
                for num in idx_list:
                    rows[-1][num] += row[num]
            elif len(idx_list) == 1:
                for num in idx_list:
                    rows[-1][num] += row[num]
            else:
                rows.append(row)
        else:
            rows.append(row)

    return headers, rows


# Function to convert parsed table to JSON
def convert_mathpix_to_json(tsv_string, latex_input):
    headers, rows = parse_tsv(tsv_string)
    merged_dict = defaultdict(list)
    for row in rows:
        for header, value in zip(headers, row):
            merged_dict[header].append(value)
    full_table = {}
    full_table["table_dict"] = dict(merged_dict)
    return full_table


class MathPixTableStructurePredictor(ImagePredictorABC):
    def __init__(self, mathpix_headers, expansion_value=0.01):
        super().__init__(entity_to_process=TablesFieldName, find_caption=True)
        self.expand_ratio = expansion_value
        self.headers = mathpix_headers

    @property
    def predictor_identifier(self) -> str:
        return "MathPix"

    @property
    def preferred_layer_name(self) -> str:
        return f"TAGGED_IMAGE_{self.predictor_identifier}"

    def process_image(self, image) -> ImagePredictionResult:
        try:
            math_pix_input = get_mathpix_input(encode_image(image))
            response = requests.post(MATHPIX_ENDPOINT, headers=self.headers, json=math_pix_input)
            response_data = response.json()
            if "error_info" in response_data.keys():
                raise Exception(f"MathPix failed to parse a table!: {response_data['error_info']}")
            tsv_data = response_data["data"][0]["value"]
            latex_data = response_data["text"]
            json_data = convert_mathpix_to_json(tsv_data, latex_data)

            table_data = ImagePredictionResult(
                raw_prediction=tsv_data,
                predicted_dict=json_data,
            )

            return table_data
        except Exception as e:
            raise e

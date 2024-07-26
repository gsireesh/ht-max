from typing import List
import io
import torch
from torchvision import transforms
from transformers import TableTransformerForObjectDetection

from papermage import Box, Document, Entity, TablesFieldName
from papermage.predictors import BasePredictor
from papermage_components.utils import get_table_image, get_text_in_box, globalize_bbox_coordinates

import requests
import base64
import json
import csv
from collections import defaultdict
from PIL import Image

def get_mathpix_input(encoded_image):
    json_data = {
        "src": f"data:image/jpeg;base64,{encoded_image}",
        "formats": ["text", "data"],
        "data_options": {
            "include_tsv": True,
        }
    }
    return json_data


def get_nearby_captions(table, doc, expansion_factor):
    box = table.boxes[0]

    exp_h = expansion_factor * box.h
    diff_h = exp_h - box.h

    search_box = Box(l=box.l, t=box.t - diff_h / 2, w=box.w, h=exp_h, page=box.page)
    potential_captions = doc.find(query=search_box, name="captions")
    return potential_captions


class TableStructurePredictor(BasePredictor):
    def __init__(self,app_url,mathpix_headers,expandsion_value = 0.01):
        self.expand_ratio = expandsion_value
        self.url = app_url
        self.headers = mathpix_headers
      
    def _predict(self, doc: Document) -> List[Entity]:

        for table in getattr(doc, TablesFieldName):
            if abs(table.spans[0].start - table.spans[0].end)<100:
                continue
            table_image = get_table_image(table, doc,None,self.expand_ratio)
            math_pix_input = get_mathpix_input(encode_image(table_image))
            response = requests.post(self.url, headers=self.headers, json=math_pix_input)
            response_data = response.json()
            if "error_info" in response_data.keys():
                print("get an error")
                continue
            tsv_data = response_data["data"][0]["value"]
            latex_data = response_data["text"]
            json_data = convert_mathpix_to_json(tsv_data,latex_data)

            table.metadata["table_dict"] = json_data["table_dict"]
            candidate_table_captions = get_nearby_captions(table, doc, expansion_factor=2)
            if candidate_table_captions:
                if len(candidate_table_captions) > 1:
                    best_candidate = None
                    for caption in candidate_table_captions:
                        if min(abs(caption.spans[0].start-table.spans[0].end),abs(caption.spans[0].end-table.spans[0].start)) == 1:
                            best_candidate = caption
                            break
                else:
                    best_candidate = candidate_table_captions[0]
                table.metadata["caption_id"] = best_candidate.id
                table.metadata["caption"] = best_candidate.text

            # n+=1

        return []

def encode_image(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    image_bytes = buffer.read()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    return encoded_image

def find_non_empty_indices(lst):
    return [index for index, value in enumerate(lst) if value != ""]
# Function to parse TSV data
def parse_caption_note(latex_input):
    parts = latex_input.split(r'\begin{tabular}')
    caption = parts[0].strip()
    post_table_content = parts[-1].split(r'\end{tabular}')[-1].strip()
    return caption, post_table_content

def parse_tsv(tsv_string):
    # Read the TSV data
    reader = csv.reader(tsv_string.splitlines(), delimiter='\t')
    
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
def convert_mathpix_to_json(tsv_string,latex_input):
    headers, rows = parse_tsv(tsv_string)
    merged_dict = defaultdict(list)
    for row in rows:
        for header, value in zip(headers, row):
            merged_dict[header].append(value)
    full_table = {}
    full_table["table_dict"] = dict(merged_dict)
    return full_table

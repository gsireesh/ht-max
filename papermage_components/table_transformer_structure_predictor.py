from dataclasses import dataclass

import torch
from torchvision import transforms
from transformers import TableTransformerForObjectDetection
from transformers.models.table_transformer.modeling_table_transformer import (
    TableTransformerObjectDetectionOutput as TatrOutput,
)

from papermage import Box, Document, Entity, TablesFieldName
from papermage_components.interfaces import ImagePredictionResult, ImagePredictorABC
from papermage_components.utils import get_table_image, get_text_in_box, globalize_box_coordinates


@dataclass
class TatrPrediction:
    label: str
    score: float
    bbox: Box


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image


def box_cxcywh_to_cornerwh(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), w, h]
    return torch.stack(b, dim=1)


def format_model_output(outputs: TatrOutput, id2label: dict[int, str]) -> list[TatrPrediction]:
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in box_cxcywh_to_cornerwh(pred_bboxes)]

    cell_info = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            cell_info.append(
                TatrPrediction(label=class_label, score=float(score), bbox=Box(*bbox, -1))
            )

    return cell_info


structure_transform = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Function to find cell coordinates
def find_cell_coordinates(row: TatrPrediction, column: TatrPrediction):
    cell_bbox = Box(column.bbox.l, row.bbox.t, column.bbox.w, row.bbox.h, -1)
    return cell_bbox


def get_header_column_cell_mapping(
    predictions: list[TatrPrediction],
) -> list[tuple[Box, list[Box]]]:
    # Extract rows and columns
    rows = [entry for entry in predictions if entry.label == "table row"]
    columns = [entry for entry in predictions if entry.label == "table column"]
    column_headers = [entry for entry in predictions if entry.label == "table column header"]

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x.bbox.t)
    columns.sort(key=lambda x: x.bbox.l)

    if not column_headers:
        return []

    table_representation = []
    for j, column in enumerate(columns):
        column_heading = find_cell_coordinates(column_headers[0], column)
        column_boxes = []
        for i, row in enumerate(rows):
            if i == 0:
                continue
            cell_bbox = find_cell_coordinates(row, column)
            column_boxes.append(cell_bbox)
        table_representation.append((column_heading, column_boxes))

    return table_representation


def shrink_box(box, w_shrink_factor, h_shrink_factor):
    new_width = w_shrink_factor * box.w
    new_height = h_shrink_factor * box.h
    width_diff = box.w - new_width
    height_diff = box.h - new_height

    return Box(box.l + (width_diff / 2), box.t + (height_diff / 2), new_width, new_height, box.page)


def convert_table_mapping_to_boxes_and_text(
    header_to_column_mapping: list[tuple[Box, list[Box]]],
    table_entity: Entity,
    doc: Document,
    w_shrink: float,
    h_shrink: float,
):
    table_text_repr = {}
    all_cell_boxes = []

    for header_cell, row_cells in header_to_column_mapping:
        table_box = table_entity.boxes[0]
        header_box = shrink_box(header_cell, w_shrink, h_shrink)
        header_box.page = table_entity.boxes[0].page

        all_cell_boxes.append(header_box)
        header_text = get_text_in_box(globalize_box_coordinates(header_box, table_box, doc), doc)

        table_text_repr[header_text] = []
        for a_cell in row_cells:
            cell_box = shrink_box(a_cell, w_shrink, h_shrink)
            cell_box.page = table_entity.boxes[0].page
            all_cell_boxes.append(cell_box)
            table_text_repr[header_text].append(
                get_text_in_box(globalize_box_coordinates(cell_box, table_box, doc), doc)
            )

    return all_cell_boxes, table_text_repr


class TableTransformerStructurePredictor(ImagePredictorABC):
    def __init__(self, model, device, w_shrink=0.95, h_shrink=0.5):
        super().__init__(TablesFieldName)
        self.model = model.to(device)
        self.w_shrink = w_shrink
        self.h_shrink = h_shrink

    def get_table_structure(
        self,
        table_image,
    ):
        pixel_values = structure_transform(table_image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(pixel_values)
        structure_id2label = self.model.config.id2label
        structure_id2label[len(structure_id2label)] = "no object"
        predictions = format_model_output(outputs, structure_id2label)
        header_column_mapping = get_header_column_cell_mapping(predictions)

        return header_column_mapping

    @classmethod
    def from_model_name(
        cls, model_name="microsoft/table-structure-recognition-v1.1-all", device="cpu"
    ):
        model = TableTransformerForObjectDetection.from_pretrained(model_name)
        return cls(model, device)

    @property
    def preferred_layer_name(self) -> str:
        return f"TAGGED_IMAGE_Table_Transformer"

    @property
    def predictor_identifier(self) -> str:
        return "Table Transformer Structure Predictor"

    def process_entity(self, table_entity: Entity) -> ImagePredictionResult:
        doc = table_entity.layer.doc
        table_image = get_table_image(table_entity, doc, expand_box_by=0)
        header_to_column_mapping = self.get_table_structure(table_image)

        table_boxes, table_dict = convert_table_mapping_to_boxes_and_text(
            header_to_column_mapping, table_entity, doc, self.w_shrink, self.h_shrink
        )
        result = ImagePredictionResult(
            raw_prediction={},
            predicted_boxes=table_boxes,
            predicted_dict=table_dict,
        )
        return result

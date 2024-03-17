from typing import List

import torch
from torchvision import transforms
from transformers import TableTransformerForObjectDetection

from papermage import Box, Document, Entity, TablesFieldName
from papermage.predictors import BasePredictor
from papermage_components.utils import get_text_in_box, globalize_bbox_coordinates


def get_table_image(table_entity: Entity, doc: Document):
    if len(table_entity.boxes) > 1:
        raise AssertionError("Table has more than one box!!")
    box = table_entity.boxes[0]
    page_image = doc.pages[box.page].images[0].pilimage
    page_w, page_h = page_image.size
    table_image = page_image.crop(
        (
            box.l * page_w,
            box.t * page_h,
            (box.l + box.w) * page_w,
            (box.t + box.h) * page_h,
        )
    )
    return table_image


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects


structure_transform = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Function to find cell coordinates
def find_cell_coordinates(row, column):
    cell_bbox = [column["bbox"][0], row["bbox"][1], column["bbox"][2], row["bbox"][3]]
    return cell_bbox


def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry["label"] == "table row"]
    columns = [entry for entry in table_data if entry["label"] == "table column"]
    column_headers = [entry for entry in table_data if entry["label"] == "table column header"]

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x["bbox"][1])
    columns.sort(key=lambda x: x["bbox"][0])

    if not column_headers:
        return {}

    table_representation = {}
    for j, column in enumerate(columns):
        column_heading = tuple(find_cell_coordinates(column_headers[0], column))
        table_representation[column_heading] = []
        for i, row in enumerate(rows):
            if i == 0:
                continue
            cell_bbox = find_cell_coordinates(row, column)
            table_representation[column_heading].append(cell_bbox)

    return table_representation


def shrink_box(box, shrink_factor):
    new_width = shrink_factor * box.w
    new_height = shrink_factor * box.h
    width_diff = box.w - new_width
    height_diff = box.h - new_height

    return Box(box.l + width_diff / 2, box.t + height_diff / 2, new_width, new_height, box.page)


def convert_table_mapping_to_boxes_and_text(header_to_column_mapping, table_entity, doc, shrink):
    table_text_repr = {}
    all_cell_boxes = []

    for header_cell, row_cells in header_to_column_mapping.items():
        table_box = table_entity.boxes[0]
        header_box = shrink_box(globalize_bbox_coordinates(header_cell, table_box, doc), shrink)
        all_cell_boxes.append(header_box)
        header_text = get_text_in_box(header_box, doc)
        table_text_repr[header_text] = []
        for a_cell in row_cells:
            cell_box = shrink_box(globalize_bbox_coordinates(a_cell, table_box, doc), shrink)
            all_cell_boxes.append(cell_box)
            table_text_repr[header_text].append(get_text_in_box(cell_box, doc))
    return all_cell_boxes, table_text_repr


class TableStructurePredictor(BasePredictor):
    def __init__(self, model, device, shrink=0.9):
        self.model = model.to(device)
        self.shrink = shrink

    @classmethod
    def from_model_name(
        cls, model_name="microsoft/table-structure-recognition-v1.1-all", device="cpu"
    ):
        model = TableTransformerForObjectDetection.from_pretrained(model_name)
        return cls(model, device)

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [TablesFieldName]

    def _predict(self, doc: Document) -> List[Entity]:
        for table in getattr(doc, TablesFieldName):

            table_image = get_table_image(table, doc)
            header_to_column_mapping = self.get_table_structure(table_image)

            table_boxes, table_dict = convert_table_mapping_to_boxes_and_text(
                header_to_column_mapping, table, doc, self.shrink
            )

            table.metadata["cell_boxes"] = table_boxes
            table.metadata["table_dict"] = table_dict

        return []

    def get_table_structure(
        self,
        table_image,
    ):
        pixel_values = structure_transform(table_image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(pixel_values)
        structure_id2label = self.model.config.id2label
        structure_id2label[len(structure_id2label)] = "no object"
        raw_cells = outputs_to_objects(outputs, table_image.size, structure_id2label)
        cell_bbox_structure = get_cell_coordinates_by_row(raw_cells)

        return cell_bbox_structure

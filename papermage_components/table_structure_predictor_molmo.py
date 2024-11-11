# import sys
# import os
import pandas as pd
from papermage import TablesFieldName
from papermage_components.interfaces.image_predictor import ImagePredictionResult, ImagePredictorABC
#from interfaces.image_predictor import ImagePredictionResult, ImagePredictorABC

from PIL import Image
import io
import json
import base64
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

table_structure_prediction_prompt = """\
Extract any tables with obvious table structure from this image as structured JSON data.
Don't try to extract any non-existent table with only "table" keyword in the text! Don't consider the authorship claim as table!
For all recognized tables, extract all data, including headers and values. If no tables are found, return an empty list.

Format the extracted data as a list of dictionaries, where each dictionary represents a table with the following structure:
{
    "title": "The full title of the table",
    "headers": ["Column1", "Column2", ...],
    "data": [
        ["Row1Col1", "Row1Col2", ...],
        ["Row2Col1", "Row2Col2", ...],
        ...
    ]
}"""

class MolmoTableStructurePredictor(ImagePredictorABC):
    def __init__(self, molmo_model=None, device=None):
        super().__init__(entity_to_process=TablesFieldName, find_caption=True)

        if molmo_model is None:
            molmo_model = 'allenai/Molmo-7B-O-0924'
        if device is None:
            device = torch.device('cpu')

        self.processor = AutoProcessor.from_pretrained(
            molmo_model,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            molmo_model,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        self.model.to(device)
        
        self.raw_table_data = None
        self.json_table_data = None
        self.table_title = None
        self.raw_response = None

    @property
    def predictor_identifier(self) -> str:
        return "Molmo"

    @property
    def preferred_layer_name(self) -> str:
        return f"TAGGED_IMAGE_{self.predictor_identifier}"
    
    def process_image(self, image) -> ImagePredictionResult:

        try:
            self.predict_table_structure(image)
            self.get_json_table_data()

            # inputs = self.processor.process()
            # #     images=[image.raw)],
            # #     text="Write a short rhyming poem about this image"
            # # )

            table_data = ImagePredictionResult(
                    raw_prediction=self.raw_response,
                    predicted_dict=self.json_table_data
                )
            return table_data
            
        except Exception as e:
            raise e

        
    
    def get_json_table_data(self) -> None:
        
        headers = self.raw_table_data["headers"]
        rows = self.raw_table_data["data"]

        try:
            # Check if all rows have the same number of columns as the headers
            if any(len(row) != len(headers) for row in rows):
                self.json_table_data = None
                raise ValueError("The number of columns in rows does not match the number of headers.")
            
            # Convert the data to a pandas DataFrame
            df = pd.DataFrame(data=rows, columns=headers)
            self.json_table_data = df.to_dict()
        
        except ValueError as ve:
            print(f"Error: {ve}")
        

    def predict_table_structure(self, image) -> None: #image: PIL.Image.Image
        
        # Resize the image if it's too large
        max_size = (1600, 1600)  # Adjust these dimensions as needed
        image.thumbnail(max_size, Image.LANCZOS)
        
        response_max_length = 4096 # adjust as needed

        inputs = self.processor.process(
            images=[image],
            text=table_structure_prediction_prompt,
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=response_max_length, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )

        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        self.raw_response = generated_text

        try:
            self.raw_table_data = json.loads(generated_text)[0] # there should only be one table.
        except json.JSONDecodeError as e:
            print("Failed to parse JSON due to json decoding error. Skipping this image.")
            raise e

    

# import sys
# import os
# sys.path.append(os.path.abspath('/Users/harryzhang/Documents/CMU/MSE_NLP/Code/htmax/papermage_components'))
import pandas as pd
from papermage import TablesFieldName
from papermage_components.interfaces.image_predictor import ImagePredictionResult, ImagePredictorABC
#from interfaces.image_predictor import ImagePredictionResult, ImagePredictorABC

import anthropic
from PIL import Image
import io
import json
import base64
import numpy as np


class ClaudeTableStructurePredictor(ImagePredictorABC):
    def __init__(self,claude_model):
        super().__init__(entity_to_process=TablesFieldName, find_caption=True)
        if claude_model is None:
            self.claude_model = "claude-3-opus-20240229"
        else:
            self.claude_model = claude_model
        
        self.raw_table_data = None
        self.json_table_data = None
        self.table_title = None
        self.raw_api_response = None

    @property
    def predictor_identifier(self) -> str:
        return "Claude"

    @property
    def preferred_layer_name(self) -> str:
        return f"TAGGED_IMAGE_{self.predictor_identifier}"
    
    def process_image(self, image) -> ImagePredictionResult:

        try:
            self.get_raw_table_data_from_images(image)
            self.get_json_table_data()

            table_data = ImagePredictionResult(
                    raw_prediction=self.raw_api_response,
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
        

    def get_raw_table_data_from_images(self, image) -> None: #image: PIL.Image.Image
        
        client = anthropic.Anthropic()

        # Resize the image if it's too large
        max_size = (1600, 1600)  # Adjust these dimensions as needed
        image.thumbnail(max_size, Image.LANCZOS)
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        if len(img_byte_arr) > 5 * 1024 * 1024:  # 5MB in bytes
            #print("Image size exceeds 5MB. Skipping this image.")
            raise Exception ("Image size exceeds 5MB. Skipping this image.")

        # Encode the binary data to base64
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        full_response = ""
        chunk_size = 4096  # Adjust this value based on your needs

        message = client.messages.create(
            model=self.claude_model,
            max_tokens=4000,  # Increased max_tokens
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": """Extract any tables with obvious table structure from this image as structured JSON data. Don't try to extract any non-existent table with only "table" keyword in the text! Don't consider the authorship claim as table!
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
                        }
                    ]
                }
            ]
        )
        
        # Collect the full response
        for chunk in message.content[0].text:
            full_response += chunk

        if len(full_response) >= chunk_size:
                print(f"Claude model received {len(full_response)} characters")    
        
        self.raw_api_response = full_response #for debugging purpose
        
        try:
            self.raw_table_data = json.loads(full_response)[0]

        except json.JSONDecodeError as e:
            print("Failed to parse JSON due to json decoding error. Skipping this image.")
            raise e

    

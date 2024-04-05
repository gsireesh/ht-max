
import os
from GPT_predictor import *

from papermage.magelib import Document


import json
from typing import Dict

# def read_json_file(file_path: str) -> Dict[str, any]:
#     with open(file_path, 'r') as file:
#         json_data = json.load(file)
#     return json_data


json_file_path = "/Users/harryzhang/ht-max/data/AM_Creep_Papers_parsed/On the creep performance of the Ti‐6Al‐4V alloy processed by additive manufacturing.json"

#json_data: Dict[str, any] = read_json_file(json_file_path) 

# #print(json_data)

with open(json_file_path, "r") as f:
	json_content = json.load(f)
	doc = Document.from_json(json_content)

predictor = GPT_predictor(api_key=os.environ.get("openai_key"))


predictor._predict(doc)




# for section in doc.reading_order_sections[2:]:
# 	result = predictor.run_gpt_ner(section.text)
# 	print(result)
# 	break
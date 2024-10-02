from PIL import Image
from papermage_components.table_structure_predictor_claude import ClaudeTableStructurePredictor

image_path = "Claude_Test_Image.png"

image = Image.open(image_path)

Claude_Predictor = ClaudeTableStructurePredictor(claude_model=None)

Image_Prediction = Claude_Predictor.process_image(image)

print(Image_Prediction.predicted_dict)
import json
import os

def extract_sentences(json_file, output_dir):
    """
    Extracts 'sentences' from a JSON file and saves them into a .txt file.
    :param json_file: Path to the JSON file.
    :param output_dir: Directory where the .txt file will be saved.
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    sentences = data.get('sentences')
    if sentences is None:
        print(f"No 'sentences' key found in {json_file}")
        return
    
    if isinstance(sentences, list):
        text_content = '\n'.join(sentences)
    elif isinstance(sentences, str):
        text_content = sentences
    else:
        print(f"Unexpected format for 'sentences' in {json_file}")
        return

    base_filename = os.path.splitext(os.path.basename(json_file))[0]
    output_file = os.path.join(output_dir, f"{base_filename}.txt")

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text_content)
    print(f"Extracted sentences from {json_file} to {output_file}")

# Example usage
input_directory = '/Users/yuehengzhang/Desktop/Desktop/CMU/NLP/ht-max/data/AM_Creep_Papers_parsed'
output_directory = '/Users/yuehengzhang/Desktop/Desktop/CMU/NLP/src/data230707'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith('.json'):
        extract_sentences(os.path.join(input_directory, filename), output_directory)

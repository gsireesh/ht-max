from pathlib import Path
import os
import subprocess
import sys
import mspx.tools.al.utils_brat
import re
# Placeholder for multiprocessing if you plan to use it
# from multiprocessing import Pool

class MatIE:
    def __init__(self, NER_model_dir="", vocab_dir="", output_folder="", gpu_id="0", decode_script=""):
        # Your path to be added. Replace '/your/path/to/src' with your actual directory path
        MatIE_src = '/Users/yuehengzhang/Desktop/Desktop/CMU/NLP/ht-max/MatIE'
        # Add the path to sys.path
        if MatIE_src not in sys.path:
            sys.path.append(MatIE_src)

        self.NER_model_dir = NER_model_dir
        self.vocab_dir = vocab_dir
        self.output_folder = output_folder
        self.gpu_id = gpu_id
        self.decode_script = decode_script

    def generate_txt(self, filename, subset, start, end):
        sentences = subset.replace("\n", "  ")
        folder_name = f'{self.output_folder}{filename.replace(" ", "_")}'
        file_path = f'{folder_name}/{start}-{end}.txt'
        os.makedirs(folder_name, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(sentences)
        print(f"Sentences have been successfully written to {file_path}")

    def run_matIE(self):
        for dir_name in os.listdir(self.output_folder):
            # Rename the folder
            if os.path.isdir(dir_name):
                dir_path = os.path.join(self.output_folder, dir_name)
                input_folder = dir_path
                output_folder = dir_path

                self.process_files_multiprocess(self.NER_model_dir, self.vocab_dir, input_folder, output_folder, self.gpu_id, self.decode_script)

    def process_files_multiprocess(self, model_dir, vocab_dir, input_folder, output_folder, gpu_id, decode_script):

        env_vars = os.environ.copy()
        env_vars['MODEL_DIR'] = model_dir
        env_vars['VOCAB_DIR'] = vocab_dir
        env_vars['CUDA_VISIBLE_DEVICES'] = "" # needs to fix later
        env_vars['INPUT_DIR'] = os.path.join("../ht-max", input_folder.replace("//", "/"))
        env_vars['OUTPUT_DIR'] = os.path.join("../ht-max", output_folder.replace("//", "/"))
        env_vars["EXTRA_ARGS"] = ""
        
        subprocess.run(['chmod', '+x', decode_script], check=True)
        subprocess.run(decode_script, shell=True, env=env_vars, cwd = "/Users/yuehengzhang/Desktop/Desktop/CMU/NLP/ht-max/MatIE", check=True)

    def parse_ann_files(self,directory):
        span_list = []
        
        # Regex to extract start value from file name
        file_name_regex = re.compile(r'(\d+)-\d+\.ann')
        
        # Iterate over all .ann files in the specified directory
        for file_name in os.listdir(directory):
            if file_name.endswith('.ann'):
                match = file_name_regex.search(file_name)
                if match:
                    start_offset = int(match.group(1))
                    
                    file_path = os.path.join(directory, file_name)
                    with open(file_path, 'r') as file:
                        for line in file:
                            if line.startswith('T'):
                                parts = line.split('\t')
                                if len(parts) >= 3:
                                    entity_info = parts[1].split()
                                    if len(entity_info) >= 3:
                                        # Adjust span positions with start_offset
                                        adjusted_start = int(entity_info[1]) + start_offset
                                        adjusted_end = int(entity_info[2]) + start_offset
                                        # Add the adjusted span to the list
                                        span_list.append([adjusted_start, adjusted_end])
        
        return span_list


    # Example usage
    # matie = MatIE(NER_model_dir="path/to/model", vocab_dir="path/to/vocab", output_folder="path/to/output", gpu_id="0", decode_script="path/to/decode.sh")
    # matie.run_matIE()

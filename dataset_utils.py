"""
Simple Dataset Utilities for Bio-Medical AI Competition

This module contains only the essential CureBenchDataset class and related utilities
for loading bio-medical datasets in the competition starter kit.

Note: Data should be preprocessed using preprocess_data.py to add dataset_type fields
before using this module.
"""

import json
import os
import sys
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("Warning: PyTorch not available. Some features may not work.")
    # Create dummy classes for basic functionality
    class Dataset:
        pass
    class DataLoader:
        def __init__(self, *args, **kwargs):
            pass


def read_and_process_json_file(file_path):
    """
    Reads a JSON file and processes it into a standardized format.
    Handles both single JSON objects and line-delimited JSON files.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Try to read as line-delimited JSON first
            try:
                data = [json.loads(line) for line in file if line.strip()]
                # If first item is a list, flatten it
                if data and isinstance(data[0], list):
                    data = [item for sublist in data for item in sublist]
                return data
            except json.JSONDecodeError:
                # If that fails, try reading as single JSON object
                file.seek(0)
                content = file.read()
                data = json.loads(content)
                return data
                
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error: Unexpected error reading {file_path}: {e}")
        return []


class CureBenchDataset(Dataset):
    """
    Dataset class for FDA drug labeling data.
    
    This class handles loading and processing FDA drug labeling questions
    for the bio-medical AI competition. It supports:
    - Multiple choice questions
    - Open-ended questions  
    - Drug name extraction tasks
    - Subset filtering by FDA categories
    
    Example usage:
        dataset = CureBenchDataset("fda_data.json")
        question, options, answer = dataset[0]
    """
    
    def __init__(self, json_file):
        """
        Initialize the FDA Drug Dataset.
        
        Args:
            json_file (str): Path to the JSON file containing FDA data
        """
        
        # Load the data
        self.data = read_and_process_json_file(json_file)
        
        if not self.data:
            print(f"Warning: No data loaded from {json_file}")
            self.data = []
            return
        
        print(f"CureBenchDataset initialized with {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        Returns:
            - For multiple choice: (question_type, id, question, answer, meta_question, llm_category, options)
            - For open-ended: (question_type, id, question, answer, meta_question, llm_category, options)
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
            
        item = self.data[idx]
        
        # Extract basic fields
        question_type = item['question_type']
        question = item.get('question', '')
        answer = item.get('correct_answer', item.get('answer', ''))
        meta_question = ""
        id_value = item['id']
        llm_category = item.get('llm_category', 'unknown')

        ### question是LLM要回答的字段（还会放入一个template），  meta_question其实是用于抽答案的prompt
        if question_type == 'multi_choice':
            ### 1. get option_str
            options = item['options']
            options_list = '\n'.join([f"{opt}: {options[opt]}" for opt in sorted(options.keys())])
            ### 2. question
            question = f"{question}\n{options_list}"
            ### 3. meta question (为空)
            meta_question = ""
            # 确保options是直接的字符串格式，不是列表
            clean_options = {k: v if isinstance(v, str) else v[0] if isinstance(v, list) and len(v) > 0 else "" for k, v in options.items()}
            return {
                'question_type': question_type, 
                'id': id_value, 
                'question': question, 
                'answer': answer, 
                'meta_question': meta_question, 
                'llm_category': llm_category, 
                'options': clean_options
            }
            
        elif question_type == 'open_ended_multi_choice':
            ### 1. get option_str
            options = item['options']
            options_list = '\n'.join([f"{opt}: {options[opt]}" for opt in sorted(options.keys())])
            ### 2. question (no add options_list)
            question = f"{question}"
            ### 3. meta_question(下面是两种设定)
            #######>>>> 不加强制输出选项（原始设定）
            # meta_question = f"The following is a multiple choice question about medicine and the agent's open-ended answer to the question. Convert the agent's answer to the final answer format using the corresponding option label, e.g., 'A', 'B', 'C', 'D', 'E' or 'None'. \n\n\
                  # Question: {question}\n{options_list}\n\n"
            #######>>>>  加强制输出选项（原始设定）
            meta_question = f"The following is a multiple choice question about medicine and the agent's open-ended answer to the question. Convert the agent's answer to the final answer format using the corresponding option label, e.g., 'A', 'B', 'C', 'D'. \
                You MUST give the closest answer from the options with exact letter, DO NOT output None. \n\n\
                    Question: {question}\n{options_list}\n\n"
            
            return {
                'question_type': question_type, 
                'id': id_value, 
                'question': question, 
                'answer': answer, 
                'meta_question': meta_question, 
                'llm_category': llm_category, 
                'options': options
            }
        
        elif question_type == 'open_ended':
            question = f"{question}"
            ### 3. meta_question(下面为空)
            meta_question = ""
            # 对于open_ended类型，options可能为空或不存在
            options = item.get('options', {})
            # 确保options是直接的字符串格式，不是列表
            clean_options = {k: v if isinstance(v, str) else v[0] if isinstance(v, list) and len(v) > 0 else "" for k, v in options.items()}
            return {
                "question_type": question_type,
                "id": id_value,
                "question": question,
                "answer": answer,
                "meta_question": meta_question,
                "llm_category": llm_category,
                "options": clean_options
            }
            
        else:
            raise ValueError(f"Unsupported question type: {question_type}")

def build_dataset(dataset_path=None):
    """
    Build a dataset based on the dataset name and configuration.
    
    This is the main function used by the competition framework to load datasets.
    
    Args:
        dataset_name (str): Name of the dataset ('yesno', 'treatment', or FDA subset name)
        dataset_path (str): Path to the dataset file
        
    Returns:
        Dataset: Configured dataset object
    """
    print("dataset_path:", dataset_path)
    dataset = CureBenchDataset(dataset_path)
    return dataset
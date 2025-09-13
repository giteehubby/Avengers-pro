from dataset_utils import CureBenchDataset
from models import APIModel
from get_val_answer import extract_multiple_choice_answer, parse_boxed_answer_or_call_llm
import json
import os
import csv
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def process_single_question(
    question_data: Dict,
    model: APIModel,
    writer_lock: threading.Lock,
    writer: csv.DictWriter
) -> None:
    question_id = question_data['id']
    question_type = question_data['question_type']

    # if id with the model already in submissions/submission.csv

    with open('results/id2model_map.json', 'r', encoding='utf-8') as f:
        id2model_map = json.load(f)
    if question_id in id2model_map and id2model_map[question_id] == model.model_name:
        with open('submissions/submission.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['id'] == question_id:
                    writer.writerow(row)
                    return

    # Generate answer
    if question_type == 'multi_choice':
        prompt = f"The following is a multiple-choice question about medicine. Question: {question_data['question']}\n\nFirst, please think through the question carefully, considering the relevant knowledge and possible reasoning. Then, provide your final answer in the format \\boxed{{}}."
    else:
        prompt = f"The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question_data['question']}\n\nAnswer:"
    
    response, reasoning_trace = model.inference(prompt)
    import pdb;pdb.set_trace()
    # Extract answer based on question type
    if question_type == 'multi_choice':
        answer = parse_boxed_answer_or_call_llm(
            question_data['question'], 
            response, 
            model
        )
    else:
        answer = response.strip()
        meta_prompt = f"{question_data['meta_question']}Agent's answer: {response.strip()}\n\nMulti-choice answer:"
        meta_response, meta_reasoning = model.inference(meta_prompt)
        reasoning_trace += meta_reasoning
        answer = extract_multiple_choice_answer(meta_response)
    
    # Write to CSV with thread lock
    with writer_lock:
        writer.writerow({
            'id': question_id,
            'prediction': answer,
            'choice': answer,
            'reasoning': reasoning_trace
        })



def process_questions(
    test_dataset: CureBenchDataset, 
    model: APIModel,
    output_file: str = "submission.csv"
) -> None:
    """
        Process all questions using the routing model to select the best model,
        generate answers, and save results in submission format.
    """

    

    # Prepare output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'prediction', 'choice','reasoning']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Thread-safe writer lock
        writer_lock = threading.Lock()
        
        # Process questions in parallel with tqdm progress bar
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(
                    process_single_question,
                    test_data,
                    model,
                    writer_lock,
                    writer
                )
                for test_data in test_dataset
            ]
            
            for future in tqdm(as_completed(futures), total=len(test_dataset), desc="Processing questions"):
                future.result()

def main():
    # Load the routing model
    print("Loading routing model...")
    
    # Load validation dataset
    print("Loading validation dataset...")
    test_dataset = CureBenchDataset('raw_data/curebench_testset_phase1_with_tags.jsonl')
    print(f"Loaded {len(test_dataset)} testset questions")

    load_dotenv()
    api_url = os.getenv('OPENAI_BASE_URL')
    api_key = os.getenv('OPENAI_API_KEY')
    models = ['openai/gpt-4o', 'google/gemini-2.5-flash']
    for model in models:
        api_model = APIModel(model, api_url, api_key)
        process_questions(test_dataset, api_model, f'submissions/submission_{model.split("/")[-1]}.csv')
    
    print(f"Processing completed! Results saved to submissions/submission_{model.split('/')[-1]}.csv")

if __name__ == '__main__':
    main()

"""
python test_on_single_model.py
"""
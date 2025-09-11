from config import SimpleClusterConfig
from build_simple_cluster_router import SimpleClusterRouter
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

# Load environment variables
load_dotenv()

def load_routing_model(config_path: str = "models/val_0/config.json") -> SimpleClusterRouter:
    """Load the pre-trained routing model"""
    config = SimpleClusterConfig.from_file(config_path,'data/validation_results.jsonl')
    router = SimpleClusterRouter(config)
    
    # Load the saved cluster models
    model_dir = "models"  # Default model directory
    if hasattr(config, 'export_cluster') and config.export_cluster:
        model_dir = config.export_cluster
    
    router = SimpleClusterRouter.from_saved_models(config, model_dir)
    return router

def process_single_question(
    question_data: Dict,
    models: Dict,
    selected_models: List[str],
    available_models: List[str],
    writer_lock: threading.Lock,
    writer: csv.DictWriter
) -> None:
    question_id = question_data['id']
    question_type = question_data['question_type']
    
    # Route the question to get best model(s)
    if not selected_models:
        print(f"\033[31mWarning: No models selected for question {question_id}\033[0m")
        selected_models = available_models[:1]  # Fallback to first available model
    
    # Use the first selected model (can be extended to use multiple)
    model_name = selected_models[0]
    model = models[model_name]
    
    # Generate answer
    if question_type == 'multi_choice':
        prompt = f"The following is a multiple-choice question about medicine. Question: {question_data['question']}\n\nFirst, please think through the question carefully, considering the relevant knowledge and possible reasoning. Then, provide your final answer in the format \\boxed{{}}."
    else:
        prompt = f"The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question_data['question']}\n\nAnswer:"
    
    response, reasoning_trace = model.inference(prompt)

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
    
    return f"Processed question ID: {question_id} with model {model_name}"


def process_questions_with_routing(
    test_dataset: CureBenchDataset, 
    router: SimpleClusterRouter,
    output_file: str = "submission.csv"
) -> None:
    """
    Process all questions using the routing model to select the best model,
    generate answers, and save results in submission format.
    """
    # Initialize API models
    api_url = os.getenv('OPENAI_BASE_URL')
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Get all available models from the router
    available_models = router.available_models
    models = {model: APIModel(model, api_url, api_key) for model in available_models}
    
    # Prepare output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'prediction', 'choice','reasoning']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Thread-safe writer lock
        writer_lock = threading.Lock()
        # import pdb; pdb.set_trace()
        selected_models_list = router.route_queries_batch([question_data['question'] for question_data in test_dataset])
        # import pdb; pdb.set_trace()
        # Process questions in parallel with tqdm progress bar
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(
                    process_single_question,
                    test_dataset[i],
                    models,
                    selected_models,
                    available_models,
                    writer_lock,
                    writer
                )
                for i,selected_models in enumerate(selected_models_list)
            ]
            
            for future in tqdm(as_completed(futures), total=len(test_dataset), desc="Processing questions"):
                result = future.result()
                # print(result)

def main():
    # Load the routing model
    print("Loading routing model...")
    router = load_routing_model()
    
    # Load validation dataset
    print("Loading validation dataset...")
    test_dataset = CureBenchDataset('raw_data/curebench_testset_phase1_with_tags.jsonl')
    print(f"Loaded {len(test_dataset)} testset questions")
    
    # Process questions and save results
    print("Processing questions with routing...")
    process_questions_with_routing(test_dataset, router)
    
    print("Processing completed! Results saved to submission.csv")

if __name__ == '__main__':
    main()

"""
python test_simple_cluster_router.py 
"""
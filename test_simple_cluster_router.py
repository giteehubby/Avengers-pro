from config import SimpleClusterConfig
from simple_cluster_router import SimpleClusterRouter
from dataset_utils import CureBenchDataset
from models import APIModel
from get_val_answer import extract_multiple_choice_answer, parse_boxed_answer_or_call_llm
import json
import os
import csv
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_routing_model(config_path: str = "config/simple_config.json") -> SimpleClusterRouter:
    """Load the pre-trained routing model"""
    config = SimpleClusterConfig.from_file(config_path)
    router = SimpleClusterRouter(config)
    
    # Load the saved cluster models
    model_dir = "models"  # Default model directory
    if hasattr(config, 'export_cluster') and config.export_cluster:
        model_dir = config.export_cluster
    
    router = SimpleClusterRouter.from_saved_models(config, model_dir)
    return router

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
        fieldnames = ['id', 'question_type', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each question
        for i in range(len(test_dataset)):
            question_data = test_dataset[i]
            question_id = question_data['id']
            question_type = question_data['question_type']
            
            # Route the question to get best model(s)
            selected_models = router.route_queries_batch([question_data['question']])[0]
            
            if not selected_models:
                print(f"Warning: No models selected for question {question_id}")
                selected_models = available_models[:1]  # Fallback to first available model
            
            # Use the first selected model (can be extended to use multiple)
            model_name = selected_models[0]
            model = models[model_name]
            
            # Generate answer
            if question_type == 'multi_choice':
                prompt = f"The following is a multiple-choice question about medicine. Question: {question_data['question']}\n\nFirst, please think through the question carefully, considering the relevant knowledge and possible reasoning. Then, provide your final answer in the format \\boxed{{}}."
            else:
                prompt = f"The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question_data['question']}\n\nAnswer:"
            
            response, _ = model.inference(prompt)
            
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

                ### 再规则提取
                answer = extract_multiple_choice_answer(meta_response)
            
            # Write to CSV
            writer.writerow({
                'id': question_id,
                'question_type': question_type,
                'answer': answer
            })
            
            print(f"Processed question {i+1}/{len(test_dataset)} (ID: {question_id}) with model {model_name}")

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











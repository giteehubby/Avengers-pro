#!/usr/bin/env python3
import json
import os
import re
import time
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataset_utils import CureBenchDataset
from models import APIModel
from tqdm import tqdm
import argparse    

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()



api_url = os.getenv('OPENAI_BASE_URL')
api_key = os.getenv('OPENAI_API_KEY')

def extract_multiple_choice_answer(response: str) -> str:
    """从模型响应中提取字母答案"""
    if not response or response is None:
        return ""
        
    response = response.strip().upper()
    
    # 查找开头的字母
    # if response and response[0] in ['A', 'B', 'C', 'D', 'E']:
    if response and response[0] in ['A', 'B', 'C', 'D']:
        return response[0]
    tqdm.write('Warning: the answer is not in the format A, B, C, or D.')
    
    patterns = [
        r"(?:answer is|answer:|is)\s*([ABCD])",
        r"([ABCD])\)",
        r"\b([ABCD])\b"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    tqdm.write('\033[31mextract failed,default to A\033[0m')
    return "A"



def parse_boxed_answer_or_call_llm(question: str, response_with_think_and_boxed: str,llm:APIModel) -> str:
    # 格式： ..think..\boxed{}
    # parse boxed answer
    import re
    match = re.search(r'\\boxed{(.*?)}', response_with_think_and_boxed)
    if match:
        return match.group(1)
        
        # call llm
    meta_question = f"""The following is a multiple choice question about medicine and the agent's open-ended answer to the question. Convert the agent's answer to the final answer format using the corresponding option label, e.g., 'A', 'B', 'C', 'D'.
        Question: {question}
        Agent's answer: {response_with_think_and_boxed.strip()}
        Multi-choice answer:
        """
    meta_response, meta_reasoning = llm.inference(meta_question)
    choice = extract_multiple_choice_answer(meta_response)
    return choice

def process_question(question_data: dict, model: APIModel) -> Tuple[str, str, float]:
    """
    Process a single question with a specific model.
    Returns (question_id, predicted_choice, performance_score)
    """
    # import pdb; pdb.set_trace()
    if question_data['question_type'] == 'multi_choice':
        prompt = f"The following is a multiple-choice question about medicine. Question: {question_data['question']}\n\nFirst, please think through the question carefully, considering the relevant knowledge and possible reasoning. Then, provide your final answer in the format \\boxed{{}}."
    elif question_data['question_type'] in ["open_ended_multi_choice", "open_ended"]:
        prompt = f"The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question_data['question']}\n\nAnswer:"
    else:
        raise ValueError(f"Invalid question type: {question_data['question_type']}")

    
    response, reasoning_trace = model.inference(prompt)
    # import pdb; pdb.set_trace()
    
    # 初始化预测字典
    prediction = {
        "choice": "",
        "open_ended_answer": ""
    }

    prediction["open_ended_answer"] = response.strip()

    if question_data['question_type'] == 'multi_choice':
        prediction['choice'] = parse_boxed_answer_or_call_llm(question_data['question'], response, model)
        prediction["open_ended_answer"] = response.strip()
        performance = 1.0 if prediction["choice"] == question_data["answer"] else 0.0
    elif question_data['question_type'] == "open_ended_multi_choice":
        prediction["open_ended_answer"] = response.strip()
        # before
        ### 先LLM提取
        meta_prompt = f"{question_data['meta_question']}Agent's answer: {response.strip()}\n\nMulti-choice answer:"
        meta_response, meta_reasoning = model.inference(meta_prompt)
        reasoning_trace += meta_reasoning

        ### 再规则提取
        choice = extract_multiple_choice_answer(meta_response)
        prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""

        performance = 1.0 if prediction["choice"] == question_data["answer"] else 0.0
    # elif question_data['question_type'] == "open_ended":
    #     prediction["open_ended_answer"] = response.strip()
    #     performance = 1.0 if prediction["open_ended_answer"] == question_data["answer"] else 0.0
    else:
        raise ValueError(f"Invalid question type: {question_data['question_type']}")

    return prediction, reasoning_trace,performance


def get_model_answers(val_dataset: CureBenchDataset, models: List[str], output_file: str = "data/val_results.jsonl", reasoning_output_file: str = "data/val_reasoning_traces.jsonl"):
    """
    Process all questions in the validation dataset with specified models and save results.
    Also saves reasoning traces to a separate file.
    """
    print(f"Processing {len(val_dataset)} questions with {len(models)} models...")
    
    # Dictionary to store all results
    all_results = {}
    # List to store reasoning traces
    reasoning_traces = []


    
    # Process each question
    for i in range(len(val_dataset)):
        print(f"\nProcessing question {i+1}/{len(val_dataset)}")
        
        # Get question data
        question_data = val_dataset[i]
        # import pdb; pdb.set_trace()

        
        print(f"  Question ID: {question_data['id']}")
        print(f"  Question Type: {question_data['question_type']}")
        print(f"  Category: {question_data['llm_category']}")
        
        # Initialize result record for this question
        if question_data['id'] not in all_results:
            all_results[question_data['id']] = {
                "query": question_data['question'],
                "records": {},
                "usages": {},
                "dataset": "cure_bench_phase_1_val",
                "index": question_data['id'],
                "question_type": question_data['question_type'],
                "correct_answer": question_data['answer'],
                "llm_category": question_data['llm_category']
            }
        
        # Process with each model
        model_results = {}

        for model in models:
            try:
                api_model = APIModel(model, api_url, api_key)
                

                prediction, reasoning_trace,performance = process_question(question_data, api_model)
                # import pdb; pdb.set_trace()
                model_results[model] = {
                    "prediction": prediction,
                    "performance": performance
                }
                all_results[question_data['id']]["records"][model] = performance
                all_results[question_data['id']]["usages"][model] = {"cost": 0.0}
                
                # Store reasoning trace
                reasoning_traces.append({
                    "question_id": question_data['id'],
                    "question": question_data['question'],
                    "correct_answer": question_data['answer'],
                    "model": model,
                    "reasoning": reasoning_trace,
                    "prediction": prediction,
                    "performance": performance
                })
                
                tqdm.write(f"    {model}: {prediction} (correct: {question_data['answer']}, score: {performance})")
                
            except Exception as e:
                tqdm.write(f"    Error with {model}: {e}")
                model_results[model] = {
                    "prediction": "ERROR",
                    "performance": 0.0
                }
                all_results[question_data['id']]["records"][model] = 0.0
                all_results[question_data['id']]["usages"][model] = {"cost": 0.0}
                
                # Store error in reasoning trace
                reasoning_traces.append({
                    "question_id": question_data['id'],
                    "question": question_data['question'],
                    "correct_answer": question_data['answer'],
                    "model": model,
                    "reasoning": f"Error: {str(e)}",
                    "prediction": "ERROR",
                    "performance": 0.0
                })
        
        # Save intermediate results every 10 questions
        if (i + 1) % 10 == 0:
            save_results(all_results, output_file + f".tmp_{i+1}")
            save_reasoning_traces(reasoning_traces, reasoning_output_file + f".tmp_{i+1}")

    # Calculate and display summary statistics
    print_summary_stats(all_results, models)

    # Save final results
    save_results(all_results, output_file)
    save_reasoning_traces(reasoning_traces, reasoning_output_file)
    
    return all_results, reasoning_traces

def save_results(results_dict: Dict, output_file: str):
    """
    Save results in the same format as convert_data.py
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert to list format
    results_list = list(results_dict.values())
    
    # Write to JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in results_list:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    tqdm.write(f"\\nResults saved to {output_file}")
    tqdm.write(f"Total records: {len(results_list)}")

def print_summary_stats(results_dict: Dict, models: List[str]):
    """
    Print summary statistics for all models
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for model in models:
        correct_count = 0
        total_count = 0
        
        for result in results_dict.values():
            if model in result["records"]:
                performance = result["records"][model]
                if performance == 1.0:
                    correct_count += 1
                total_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"{model}: {accuracy:.4f} ({correct_count}/{total_count})")

def save_reasoning_traces(reasoning_traces_list: List[Dict], output_file: str):
    """
    Save reasoning traces in JSONL format.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for trace in reasoning_traces_list:
            f.write(json.dumps(trace, ensure_ascii=False) + '\n')
    tqdm.write(f"\nReasoning traces saved to {output_file}")
    tqdm.write(f"Total reasoning traces: {len(reasoning_traces_list)}")

def get_model_answers_parallel(val_dataset: CureBenchDataset, models: List[str], output_file: str = "data/val_results.jsonl", reasoning_output_file: str = "data/val_reasoning_traces.jsonl", max_workers: int = 4):
    """
    并行版本：处理所有问题并保存结果，使用tqdm显示进度
    """
    print(f"Processing {len(val_dataset)} questions with {len(models)} models in parallel...")
    print(f"Max workers: {max_workers}")
    
    # Dictionary to store all results
    all_results = {}
    # List to store reasoning traces
    reasoning_traces = []
    
    # 线程锁，用于保护共享数据结构
    results_lock = Lock()
    traces_lock = Lock()
    
    def process_single_question_model(question_data: Dict, model: str) -> Dict:
        """处理单个问题-模型组合"""
        try:
            # import pdb; pdb.set_trace()
            api_model = APIModel(model, api_url, api_key)
            prediction, reasoning_trace, performance = process_question(question_data, api_model)
            
            return {
                'question_id': question_data['id'],
                'model': model,
                'prediction': prediction,
                'performance': performance,
                'reasoning_trace': reasoning_trace,
                'success': True
            }
        except Exception as e:
            return {
                'question_id': question_data['id'],
                'model': model,
                'prediction': "ERROR",
                'performance': 0.0,
                'reasoning_trace': f"Error: {str(e)}",
                'success': False
            }
    
    # 准备所有任务
    tasks = []
    for i in range(len(val_dataset)):
        question_data = val_dataset[i]
        question_id = question_data['id']
        question_type = question_data['question_type']
        if question_type == "open_ended":
            # 'open_ended'类型不参与积分
            continue
        
        # 初始化结果记录
        if question_id not in all_results:
            all_results[question_id] = {
                "query": question_data['question'],
                "records": {},
                "usages": {},
                "dataset": "cure_bench_phase_1_val",
                "index": question_id,
                "question_type": question_type,
                "correct_answer": question_data['answer'],
                "llm_category": question_data['llm_category']
            }
        
        # 为每个模型创建任务
        for model in models:
            tasks.append((question_data, model))
    
    # 使用tqdm和ThreadPoolExecutor并行处理
    total_tasks = len(tasks)
    completed_tasks = 0
    save_interval = max(1, total_tasks // 10)  # 每10%保存一次
    # import pdb; pdb.set_trace()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_question_model, q_data, model): (q_data, model) 
                         for q_data, model in tasks}
        
        # 使用tqdm显示进度
        if tqdm is not None:
            progress_bar = tqdm(total=total_tasks, desc="Processing questions")
        else:
            progress_bar = None
            print(f"Processing {total_tasks} tasks in parallel...")
        
        for future in as_completed(future_to_task):
            question_data, model = future_to_task[future]
            
            try:
                result = future.result()
                question_id = result['question_id']
                
                # 使用锁保护共享数据
                with results_lock:
                    all_results[question_id]["records"][model] = result['performance']
                    all_results[question_id]["usages"][model] = {"cost": 0.0}
                
                with traces_lock:
                    reasoning_traces.append({
                        "question_id": question_id,
                        "question": question_data['question'],
                        "correct_answer": question_data['answer'],
                        "model": model,
                        "reasoning": result['reasoning_trace'],
                        "prediction": result['prediction'],
                        "performance": result['performance']
                    })
                
                completed_tasks += 1
                
                # 更新进度
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'Question': question_id,
                        'Model': model,
                        'Status': 'OK' if result['success'] else 'ERROR'
                    })
                elif completed_tasks % 10 == 0:
                    tqdm.write(f"Processed {completed_tasks}/{total_tasks} tasks...")
                
                # 每完成save_interval个任务保存一次中间结果
                if completed_tasks % save_interval == 0:
                    save_results(all_results, output_file + f".tmp_{completed_tasks}")
                    save_reasoning_traces(reasoning_traces, reasoning_output_file + f".tmp_{completed_tasks}")
                
            except Exception as e:
                tqdm.write(f"Error processing {question_data['id']} with {model}: {e}")
        
        if progress_bar is not None:
            progress_bar.close()
    
    # 计算并显示总结统计
    print_summary_stats(all_results, models)
    
    # 保存最终结果
    save_results(all_results, output_file)
    save_reasoning_traces(reasoning_traces, reasoning_output_file)
    
    return all_results, reasoning_traces

def main():

    parser = argparse.ArgumentParser(description='Process validation dataset with models')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--max-workers', type=int, default=4, help='Max workers for parallel processing')
    parser.add_argument('--serial', action='store_true', help='Use serial processing (default)')
    
    args = parser.parse_args()
    
    val_dataset = CureBenchDataset('raw_data/curebench_valset_phase1_with_tags.jsonl')
    print(f'Length of val_dataset: {len(val_dataset)}')
    
    # Define models to test
    models4route = [
        'openai/gpt-4.1',
        'openai/gpt-4o', 
        'google/gemini-2.5-flash',
        'anthropic/claude-sonnet-4',
        'qwen/qwen3-235b-a22b-2507'
    ]
    
    print(f"Models to evaluate: {models4route}")
    
    output_file = "data/validation_results.jsonl"
    reasoning_output_file = "data/validation_reasoning_traces.jsonl"
    
    if args.parallel:
        print("\nUsing parallel processing...")
        results, reasoning_traces = get_model_answers_parallel(
            val_dataset, models4route, output_file, reasoning_output_file, 
            max_workers=args.max_workers
        )
    else:
        print("\nUsing serial processing...")
        results, reasoning_traces = get_model_answers(
            val_dataset, models4route, output_file, reasoning_output_file
        )
    
    print(f"\nProcessing completed! Results saved to {output_file}")
    print(f"Reasoning traces saved to {reasoning_output_file}")
    print(f"Total questions processed: {len(results)}")

if __name__ == '__main__':
    main()

"""
# 并行处理（推荐）
python get_val_answer.py --parallel --max-workers 6

# 串行处理（原有模式）
python get_val_answer.py --serial
"""
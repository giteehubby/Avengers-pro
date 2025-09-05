#!/usr/bin/env python3
import os
import json
import pandas as pd
from pathlib import Path

def load_curebench_data(file_path):
    """加载curebench数据，建立sample_id到query和correct_answer的映射"""
    id_to_data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            id_to_data[data['id']] = {
                'query': data['question'],
                'correct_answer': data['correct_answer']
            }
    return id_to_data

def process_submission_csv(csv_path, model_name, id_to_data):
    """处理单个submission.csv文件"""
    try:
        df = pd.read_csv(csv_path)
        print(f"  CSV columns: {df.columns.tolist()}")
        
        if 'id' not in df.columns or 'choice' not in df.columns:
            print(f"  Missing required columns (id, choice) in {csv_path}")
            return {}
        
        samples_data = {}
        correct_count = 0
        total_count = 0
        
        for _, row in df.iterrows():
            sample_id = row['id']
            choice = row['choice']
            
            if sample_id in id_to_data:
                correct_answer = id_to_data[sample_id]['correct_answer']
                # 比较choice和correct_answer判断正误
                performance = 1.0 if str(choice).strip() == str(correct_answer).strip() else 0.0
                samples_data[sample_id] = performance
                
                if performance == 1.0:
                    correct_count += 1
                total_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"  Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
        
        return samples_data
        
    except Exception as e:
        print(f"  Error reading CSV {csv_path}: {e}")
        return {}

def convert_to_target_format(val_dir, curebench_file, output_file):
    """将数据转换为目标格式"""
    # 加载curebench数据
    print("Loading curebench data...")
    id_to_data = load_curebench_data(curebench_file)
    print(f"Loaded {len(id_to_data)} queries")
    
    # 遍历val目录下的所有子文件夹
    all_data = []
    processed_models = 0
    
    for model_dir in os.listdir(val_dir):
        model_path = os.path.join(val_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        print(f"Processing model: {model_dir}")
        
        # 查找submission.csv文件
        csv_path = os.path.join(model_path, "submission.csv")
        
        if not os.path.exists(csv_path):
            print(f"  No submission.csv found in {model_dir}, skipping...")
            continue
        
        print(f"  Using CSV: submission.csv")
        
        try:
            # 处理submission.csv
            samples_data = process_submission_csv(csv_path, model_dir, id_to_data)
            
            if not samples_data:
                print(f"  No valid sample data found in {model_dir}, skipping...")
                continue
            
            # 为每个样本创建记录
            for sample_id, performance in samples_data.items():
                if sample_id in id_to_data:
                    record = {
                        "query": id_to_data[sample_id]['query'],
                        "records": {
                            model_dir: performance
                        },
                        "usages": {
                            model_dir: {"cost": 0.0}
                        },
                        "dataset": "cure_bench_pharse_1",
                        "index": sample_id
                    }
                    all_data.append(record)
            
            processed_models += 1
            print(f"  Processed {len(samples_data)} samples")
            
        except Exception as e:
            print(f"  Error processing {model_dir}: {e}")
            continue
    
    # 合并相同query的数据
    print(f"\nCombining data from {processed_models} models...")
    query_to_combined = {}
    
    for record in all_data:
        query = record["query"]
        sample_id = record["index"]
        
        # 使用sample_id作为唯一标识（因为每个sample_id对应唯一的query）
        key = sample_id
        
        if key not in query_to_combined:
            query_to_combined[key] = {
                "query": query,
                "records": {},
                "usages": {},
                "dataset": "cure_bench_pharse_1",
                "index": sample_id
            }
        
        # 合并records和usages
        for model, performance in record["records"].items():
            query_to_combined[key]["records"][model] = performance
        
        for model, usage in record["usages"].items():
            query_to_combined[key]["usages"][model] = usage
    
    # 转换为列表
    final_data = list(query_to_combined.values())
    
    # 写入输出文件
    print(f"Writing {len(final_data)} records to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in final_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print("Conversion completed!")
    print(f"Total records: {len(final_data)}")
    print(f"Models processed: {processed_models}")

if __name__ == "__main__":
    val_directory = "val"
    curebench_file = "curebench_valset_phase1_with_tags.jsonl"
    output_file = "data/converted_data.jsonl"
    
    convert_to_target_format(val_directory, curebench_file, output_file) 
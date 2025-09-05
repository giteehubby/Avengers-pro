#!/usr/bin/env python3
"""
聚类数量影响的快速测试脚本

用于快速测试少量聚类数量的影响，适合调试和快速验证。
"""

import subprocess
import json
import os
import sys
import matplotlib.pyplot as plt


# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_quick_test(cluster_range=[1, 2, 4, 8, 16, 32]):
    """快速测试指定的聚类数量"""
    print("快速聚类测试")
    print(f"测试聚类数量: {cluster_range}")
    
    results = []
    
    for n_clusters in cluster_range:
        print(f"\n=== 测试 {n_clusters} 个聚类 ===")
        results_path = f"results/quick_test_{n_clusters}_results.json"
        if os.path.exists(results_path):
            print(f"结果文件已存在: {results_path}")
            data = json.load(open(results_path, 'r', encoding='utf-8'))
            accuracy = data['correct_routes'] / data['total_queries']
            results.append({
                'clusters': n_clusters,
                'accuracy': accuracy,
                'correct': data['correct_routes'],
                'total': data['total_queries']
            })
            continue
        # 构建命令
        cmd = [
            "python", "simple_cluster_router.py",
            "--input", "data/converted_data.jsonl",
            "--export_cluster", f"models/quick_test_{n_clusters}/",
            "--clusters", str(n_clusters),
            "--results_path", results_path
        ]
        
        print(f"执行: {' '.join(cmd)}")
        
        try:
            
            # Windows兼容性：强制使用UTF-8编码并设置环境变量
            
            # 设置环境变量强制Python使用UTF-8编码
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='replace', env=env)
            
            
            if result.returncode == 0:
                # 读取结果
                # results_file = f"results/quick_test_{n_clusters}_results.json"
                if os.path.exists(results_path):
                    with open(results_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    accuracy = data['correct_routes'] / data['total_queries']
                    
                    
                    results.append({
                        'clusters': n_clusters,
                        'accuracy': accuracy,
                        'correct': data['correct_routes'],
                        'total': data['total_queries']
                    })
                    
                else:
                    print("结果文件未找到")
            else:
                print(f"执行失败: {result.stderr}")
                
        except Exception as e:
            print(f"错误: {e}")
    
    # 简单可视化
    if results:
        clusters = [r['clusters'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(clusters, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('聚类数量')
        plt.ylabel('准确率')
        plt.title('快速测试：聚类数量 vs 准确率')
        
        # 设置网格，x轴间距为1
        plt.grid(True, alpha=0.5, linewidth=0.8)
        plt.xticks(range(min(clusters), max(clusters) + 1, 1))  # x轴刻度间距为1
        plt.gca().grid(True, which='major', axis='x', alpha=0.7)
        plt.gca().grid(True, which='major', axis='y', alpha=0.3)
        
        # 标注数值
        for c, a in zip(clusters, accuracies):
            plt.annotate(f'{a:.1%}', (c, a), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig('results/quick_test_results.png', dpi=150)
        plt.show()
        
        # 打印结果表
        print(f"\n{'='*40}")
        print("快速测试结果汇总:")
        print(f"{'聚类数':<8} {'准确率':<8} {'正确数':<8} {'总数':<8}")
        print(f"{'-'*40}")
        for r in results:
            print(f"{r['clusters']:<8} {r['accuracy']:.1%:<8} {r['correct']:<8} {r['total']:<8}")
        
        best = max(results, key=lambda x: x['accuracy'])
        print(f"\n最佳配置: {best['clusters']} 个聚类，准确率 {best['accuracy']:.2%}")

if __name__ == "__main__":
    # 检查输入文件
    if not os.path.exists("data/converted_data.jsonl"):
        print("错误: data/converted_data.jsonl 文件不存在")
        sys.exit(1)
    
    # 运行快速测试
    run_quick_test(list(range(1,65))) 
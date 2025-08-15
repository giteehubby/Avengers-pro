#!/usr/bin/env python3
"""
Cluster Ablation Runner

Independent script for running n_clusters ablation study only.
Provides focused interface and configuration for cluster parameter optimization.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ablation.cluster_ablation import ClusterAblation
from config import SimpleClusterConfig, setup_logging




def parse_cluster_range(cluster_range_str):
    """Parse cluster range string into tuple."""
    if not cluster_range_str:
        return None
    
    parts = cluster_range_str.split(',')
    if len(parts) == 3:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    else:
        raise ValueError("Cluster range must be 'min,max,step' format")


def main():
    """Main entry point for cluster ablation."""
    parser = argparse.ArgumentParser(
        description='N-Clusters Ablation Study for Balance Cluster Router',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic cluster ablation with default range (8-80)
  python run_cluster_ablation.py --data data/dataset.json

  # Custom cluster range
  python run_cluster_ablation.py --data data/dataset.json --cluster-range 16,64,8

  # Parallel execution with progress bar (recommended)
  python run_cluster_ablation.py --data data/dataset.json --parallel --quiet

  # Custom number of workers
  python run_cluster_ablation.py --data data/dataset.json --parallel --workers 8

  # With custom configuration
  python run_cluster_ablation.py --data data/dataset.json --config custom_config.json

  # Performance-focused configuration
  python run_cluster_ablation.py --data data/dataset.json --performance-weight 0.9 --cost-sensitivity 0.1
        """
    )
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset JSON file')
    
    # Cluster ablation specific arguments
    parser.add_argument('--cluster-range', type=str,
                       help='Cluster range as "min,max,step" (e.g., "8,80,8")')
    
    # Configuration arguments
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (optional)')
    parser.add_argument('--output', type=str, default='ablation',
                       help='Output directory for results')
    
    # Parallel execution options
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel (faster execution)')
    parser.add_argument('--workers', type=int,
                       help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimize output and show progress bar only')
    
    # Output and reporting
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate markdown report after experiment')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip generating visualizations')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load base configuration
        if args.config:
            base_config = SimpleClusterConfig.from_file(args.config, args.data).to_dict()
        else:
            # Use minimal default config
            base_config = {
                'data_path': args.data,
                'seed': 42,
                'n_clusters': 40,  # Will be varied during ablation
                'performance_weight': 0.7,
                'cost_sensitivity': 0.3,
                'train_ratio': 0.7,
                'max_router': 1,
                'top_k': 1,
                'beta': 9.0,
                'max_workers': 4,
                'cluster_batch_size': 1000,
                'max_tokens': 30000,
                'embedding_model': 'Qwen3-Embedding-8B',
                'min_accuracy_threshold': 0.0,
                'budget_limit': None,
                'excluded_models': [],
                'excluded_datasets': ['aime'],
                'dataset_exclusion_mode': 'soft'
            }
        
        # Parse cluster range
        cluster_range = None
        if args.cluster_range:
            cluster_range = parse_cluster_range(args.cluster_range)
        
        logger.info(f"Starting cluster ablation study")
        logger.info(f"Dataset: {args.data}")
        logger.info(f"Cluster range: {cluster_range if cluster_range else 'default (8-80)'}")
        
        # Initialize and run cluster ablation
        cluster_ablation = ClusterAblation(output_dir=args.output)
        results = cluster_ablation.run_cluster_ablation(
            base_config=base_config,
            cluster_range=cluster_range,
            load_baseline=True,
            parallel=args.parallel,
            max_workers=args.workers,
            quiet=args.quiet
        )
        
        # Generate report if requested
        report_path = None
        if args.generate_report:
            logger.info("Generating markdown report")
            report_path = generate_cluster_report(results, args.output)
            print(f"\n📄 Report generated: {report_path}")
        
        # Print summary
        successful_experiments = results.get('successful_experiments', 0)
        total_experiments = results.get('total_experiments', 0)
        
        print(f"\n✅ Cluster ablation study completed successfully!")
        print(f"📊 Experiments: {successful_experiments}/{total_experiments} successful")
        print(f"📁 Results saved to: {args.output}/results/")
        
        if not args.no_visualizations and results.get('figure_paths'):
            print(f"📈 Visualizations saved to: {args.output}/figures/")
        
        # Show key findings
        analysis = results.get('analysis', {})
        optimal_configs = analysis.get('optimal_configurations', {})
        
        if optimal_configs.get('best_accuracy'):
            best_config = optimal_configs['best_accuracy']
            print(f"\n🎯 Key Findings:")
            print(f"   Best accuracy: n_clusters={best_config.get('n_clusters')} (accuracy: {best_config.get('accuracy', 0):.1%})")
        
        if optimal_configs.get('best_cost_efficiency'):
            eff_config = optimal_configs['best_cost_efficiency']
            print(f"   Most efficient: n_clusters={eff_config.get('n_clusters')} (efficiency: {eff_config.get('cost_efficiency', 0):.2f})")
        
        if optimal_configs.get('best_balanced'):
            balanced_config = optimal_configs['best_balanced']
            print(f"   Best balanced: n_clusters={balanced_config.get('n_clusters')} (score: {balanced_config.get('balanced_score', 0):.2f})")
        
        # Performance trends
        perf_analysis = analysis.get('performance_analysis', {})
        if perf_analysis.get('accuracy_trend'):
            trend = perf_analysis['accuracy_trend']
            print(f"   Performance trend: {trend}")
        
        # Baseline comparison
        baseline_comp = analysis.get('baseline_comparison', {})
        if baseline_comp.get('improvement_over_baseline'):
            improvement = baseline_comp['improvement_over_baseline']
            print(f"   Improvement over baseline: {improvement:.1%}")
        
    except Exception as e:
        logger.error(f"Cluster ablation study failed: {e}")
        sys.exit(1)


def generate_cluster_report(results, output_dir):
    """Generate a focused cluster ablation report."""
    report_lines = []
    
    timestamp = results.get('timestamp', datetime.now().isoformat())
    cluster_range = results.get('cluster_range', [])
    analysis = results.get('analysis', {})
    
    # Header
    report_lines.extend([
        f"# N-Clusters Ablation Study Report",
        f"",
        f"**Generated:** {timestamp}",
        f"**Experiment:** Balance Cluster Router n_clusters parameter optimization",
        f"",
        f"---",
        f""
    ])
    
    # Experiment overview
    report_lines.extend([
        f"## 📋 Experiment Overview",
        f"",
        f"**Objective:** Determine optimal number of clusters for query routing",
        f"**Parameter Varied:** n_clusters",
        f"**Range Tested:** {min(cluster_range)} to {max(cluster_range)} ({len(cluster_range)} configurations)",
        f"**Success Rate:** {results.get('successful_experiments', 0)}/{results.get('total_experiments', 0)}",
        f"",
        f"---",
        f""
    ])
    
    # Key findings
    optimal_configs = analysis.get('optimal_configurations', {})
    if optimal_configs:
        report_lines.extend([
            f"## 🎯 Key Findings",
            f""
        ])
        
        if optimal_configs.get('best_accuracy'):
            best_config = optimal_configs['best_accuracy']
            report_lines.extend([
                f"### Optimal Configuration for Accuracy",
                f"- **Best n_clusters:** {best_config.get('n_clusters')}",
                f"- **Achieved accuracy:** {best_config.get('accuracy', 0):.1%}",
                f"- **Cost efficiency:** {best_config.get('cost_analysis', {}).get('cost_efficiency', 0):.2f}",
                f""
            ])
        
        if optimal_configs.get('best_cost_efficiency'):
            eff_config = optimal_configs['best_cost_efficiency']
            report_lines.extend([
                f"### Optimal Configuration for Cost Efficiency",
                f"- **Best n_clusters:** {eff_config.get('n_clusters')}",
                f"- **Cost efficiency:** {eff_config.get('cost_efficiency', 0):.2f}",
                f"- **Accuracy:** {eff_config.get('accuracy', 0):.1%}",
                f""
            ])
        
        report_lines.extend([f"---", f""])
    
    # Performance analysis
    perf_analysis = analysis.get('performance_analysis', {})
    if perf_analysis:
        report_lines.extend([
            f"## 📈 Performance Analysis",
            f"",
            f"- **Accuracy range:** {perf_analysis.get('accuracy_range', 0):.3f}",
            f"- **Standard deviation:** {perf_analysis.get('accuracy_std', 0):.3f}",
            f"- **Performance trend:** {perf_analysis.get('accuracy_trend', 'unknown').title()}",
            f"",
            f"---",
            f""
        ])
    
    # Recommendations
    report_lines.extend([
        f"## 💡 Recommendations",
        f""
    ])
    
    if optimal_configs.get('best_accuracy'):
        best_n = optimal_configs['best_accuracy'].get('n_clusters')
        report_lines.append(f"- **For production deployment:** Use n_clusters = {best_n}")
    
    if perf_analysis.get('accuracy_trend') == 'increasing':
        max_tested = max(cluster_range)
        report_lines.append(f"- **Consider testing higher values:** Performance trend is increasing, try values > {max_tested}")
    elif perf_analysis.get('accuracy_trend') == 'decreasing':
        min_tested = min(cluster_range)
        report_lines.append(f"- **Consider testing lower values:** Performance trend is decreasing, try values < {min_tested}")
    
    report_lines.extend([
        f"- **Validate on different datasets** before final deployment",
        f"- **Monitor cost implications** when changing cluster count",
        f"",
        f"---",
        f"",
        f"*Report generated by Balance Cluster Router Ablation System*"
    ])
    
    # Save report
    report_filename = f"cluster_ablation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = Path(output_dir) / "reports" / report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return str(report_path)


if __name__ == "__main__":
    main()
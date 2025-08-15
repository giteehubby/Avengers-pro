#!/usr/bin/env python3
"""
Weight Ablation Runner

Independent script for running cost/performance weight ablation study only.
Provides focused interface for weight balance optimization.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ablation.weight_ablation import WeightAblation
from config import SimpleClusterConfig, setup_logging


def parse_weight_range(weight_range_str):
    """Parse weight range string into tuple."""
    if not weight_range_str:
        return None
    
    parts = weight_range_str.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    else:
        raise ValueError("Weight range must be 'min,max' format")


def main():
    """Main entry point for weight ablation."""
    parser = argparse.ArgumentParser(
        description='Cost/Performance Weight Ablation Study for Balance Cluster Router',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic weight ablation with default config
  python run_weight_ablation.py --data data/dataset.json

  # Custom weight configuration  
  python run_weight_ablation.py --data data/dataset.json --weight-range 0,1 --step-size 0.1

  # With custom configuration file
  python run_weight_ablation.py --data data/dataset.json --config my_config.json --step-size 0.05

  # Parallel execution
  python run_weight_ablation.py --data data/dataset.json --step-size 0.1 --parallel --quiet
        """
    )
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset JSON file')
    
    # Weight ablation specific arguments
    parser.add_argument('--weight-range', type=str, default='0,1',
                       help='Performance weight range as "min,max" (default: "0,1")')
    parser.add_argument('--step-size', type=float, default=0.1,
                       help='Step size for performance_weight iteration (default: 0.1)')
    
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
                'n_clusters': 40,
                'performance_weight': 0.7,  # Will be varied during ablation
                'cost_sensitivity': 0.3,   # Will be varied during ablation
                'train_ratio': 0.7,
                'max_router': 1,
                'top_k': 3,
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
        
        # Parse weight configuration
        performance_weight_range = parse_weight_range(args.weight_range)
        
        logger.info(f"Starting weight ablation study")
        logger.info(f"Dataset: {args.data}")
        logger.info(f"Performance weight range: {performance_weight_range}")
        logger.info(f"Step size: {args.step_size}")
        
        # Initialize and run weight ablation
        weight_ablation = WeightAblation(output_dir=args.output)
        results = weight_ablation.run_weight_ablation(
            base_config=base_config,
            performance_weight_range=performance_weight_range,
            step_size=args.step_size,
            load_baseline=True,
            parallel=args.parallel,
            max_workers=args.workers,
            quiet=args.quiet
        )
        
        # Generate report if requested
        report_path = None
        if args.generate_report:
            logger.info("Generating markdown report")
            report_path = generate_weight_report(results, args.output)
            print(f"\n📄 Report generated: {report_path}")
        
        # Print summary
        successful_experiments = results.get('successful_experiments', 0)
        total_experiments = results.get('total_experiments', 0)
        
        print(f"\n✅ Weight ablation study completed successfully!")
        print(f"📊 Experiments: {successful_experiments}/{total_experiments} successful")
        print(f"📁 Results saved to: {args.output}/results/")
        
        if not args.no_visualizations and results.get('figure_paths'):
            print(f"📈 Visualizations saved to: {args.output}/figures/")
        
        # Show key findings
        analysis = results.get('analysis', {})
        optimal_configs = analysis.get('optimal_configurations', {})
        
        if optimal_configs.get('best_accuracy'):
            best_config = optimal_configs['best_accuracy']
            perf_weight = best_config.get('performance_weight', 0)
            cost_sens = best_config.get('cost_sensitivity', 0)
            print(f"\n🎯 Key Findings:")
            print(f"   Best accuracy config: perf_weight={perf_weight:.2f}, cost_sens={cost_sens:.2f} (accuracy: {best_config.get('accuracy', 0):.1%})")
        
        if optimal_configs.get('best_cost_efficiency'):
            eff_config = optimal_configs['best_cost_efficiency']
            perf_weight = eff_config.get('performance_weight', 0)
            cost_sens = eff_config.get('cost_sensitivity', 0)
            print(f"   Most efficient config: perf_weight={perf_weight:.2f}, cost_sens={cost_sens:.2f} (efficiency: {eff_config.get('cost_efficiency', 0):.2f})")
        
        # Pareto frontier info
        pareto_analysis = analysis.get('pareto_analysis', {})
        if pareto_analysis.get('pareto_frontier_points'):
            pareto_count = pareto_analysis['pareto_frontier_points']
            print(f"   Pareto optimal configs: {pareto_count} out of {successful_experiments}")
        
    except Exception as e:
        logger.error(f"Weight ablation study failed: {e}")
        sys.exit(1)


def generate_weight_report(results, output_dir):
    """Generate a focused weight ablation report."""
    report_lines = []
    
    timestamp = results.get('timestamp', datetime.now().isoformat())
    weight_configs = results.get('weight_configurations', [])
    analysis = results.get('analysis', {})
    
    # Header
    report_lines.extend([
        f"# Cost/Performance Weight Ablation Study Report",
        f"",
        f"**Generated:** {timestamp}",
        f"**Experiment:** Balance Cluster Router weight balance optimization",
        f"",
        f"---",
        f""
    ])
    
    # Experiment overview
    report_lines.extend([
        f"## 📋 Experiment Overview",
        f"",
        f"**Objective:** Find optimal balance between performance and cost optimization",
        f"**Parameters Varied:** performance_weight, cost_sensitivity (constrained: sum = 1.0)",
        f"**Configurations Tested:** {len(weight_configs)} weight combinations",
        f"**Success Rate:** {results.get('successful_experiments', 0)}/{results.get('total_experiments', 0)}",
        f"",
        f"---",
        f""
    ])
    
    # Pareto frontier analysis
    pareto_analysis = analysis.get('pareto_analysis', {})
    if pareto_analysis:
        pareto_count = pareto_analysis.get('pareto_frontier_points', 0)
        pareto_configs = pareto_analysis.get('pareto_configurations', [])
        
        report_lines.extend([
            f"## 🎯 Pareto Frontier Analysis",
            f"",
            f"**Pareto Optimal Configurations:** {pareto_count} out of {len(weight_configs)}",
            f"",
            f"These configurations represent the best trade-offs between accuracy and cost:",
            f""
        ])
        
        for i, config in enumerate(pareto_configs[:5]):  # Show top 5
            perf_w = config.get('performance_weight', 0)
            cost_s = config.get('cost_sensitivity', 0) 
            acc = config.get('accuracy', 0)
            cost = config.get('avg_cost', 0)
            report_lines.append(f"- **Config {i+1}:** perf_weight={perf_w:.2f}, cost_sens={cost_s:.2f} → accuracy={acc:.1%}, cost=${cost:.4f}")
        
        report_lines.extend([f"", f"---", f""])
    
    # Save report
    report_filename = f"weight_ablation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = Path(output_dir) / "reports" / report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return str(report_path)


if __name__ == "__main__":
    main()
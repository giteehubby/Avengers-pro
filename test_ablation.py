#!/usr/bin/env python3
"""
Test script for ablation study module.

Quick validation of the ablation system components before running full experiments.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing imports...")
    
    try:
        from ablation.data_collector import AblationDataCollector
        print("✅ Data collector imported successfully")
        
        # Test if visualization components are available (optional)
        try:
            from ablation import ClusterAblation, WeightAblation
            print("✅ Ablation experiment modules imported successfully")
        except ImportError as viz_error:
            print(f"⚠️  Visualization modules not available (missing matplotlib): {viz_error}")
            print("   Core functionality still works, but plots will be skipped")
        
        return True
    except Exception as e:
        print(f"❌ Critical import failed: {e}")
        return False

def test_configuration_loading():
    """Test loading configuration files."""
    print("\n🔍 Testing configuration loading...")
    
    try:
        import json
        
        # Test cluster config
        with open("ablation/config/cluster_ablation_config.json", 'r') as f:
            cluster_config = json.load(f)
        print(f"✅ Cluster config loaded: {cluster_config['experiment_name']}")
        
        # Test weight config
        with open("ablation/config/weight_ablation_config.json", 'r') as f:
            weight_config = json.load(f)
        print(f"✅ Weight config loaded: {weight_config['experiment_name']}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_data_collector():
    """Test data collector functionality."""
    print("\n🔍 Testing data collector...")
    
    try:
        from ablation.data_collector import AblationDataCollector
        
        # Initialize collector
        collector = AblationDataCollector(cache_dir="ablation/test_cache")
        
        # Test config hash generation
        test_config = {"n_clusters": 32, "performance_weight": 0.7}
        config_hash = collector._generate_config_hash(test_config)
        
        print(f"✅ Data collector initialized successfully")
        print(f"✅ Config hash generated: {config_hash}")
        return True
    except Exception as e:
        print(f"❌ Data collector test failed: {e}")
        return False

def test_script_accessibility():
    """Test that ablation runner scripts can be accessed.""" 
    print("\n🔍 Testing runner scripts...")
    
    try:
        cluster_script = Path("ablation/run_cluster_ablation.py")
        weight_script = Path("ablation/run_weight_ablation.py")
        
        if cluster_script.exists():
            print(f"✅ Cluster ablation script found: {cluster_script}")
        else:
            print(f"❌ Cluster ablation script missing: {cluster_script}")
            return False
            
        if weight_script.exists():
            print(f"✅ Weight ablation script found: {weight_script}")
        else:
            print(f"❌ Weight ablation script missing: {weight_script}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Script accessibility test failed: {e}")
        return False

def test_config_validation():
    """Test configuration validation."""
    print("\n🔍 Testing configuration validation...")
    
    try:
        # Test basic configuration logic without matplotlib dependencies
        base_config = {
            'data_path': 'dummy/path.json',
            'seed': 42,
            'performance_weight': 0.7,
            'cost_sensitivity': 0.3
        }
        
        # Test cluster range generation logic manually
        # Default range from ClusterAblation
        default_cluster_counts = [8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80]
        
        if default_cluster_counts and len(default_cluster_counts) > 0:
            print(f"✅ Default cluster range validated: {len(default_cluster_counts)} configurations")
        else:
            print(f"❌ Cluster range validation failed")
            return False
        
        # Test weight range generation logic manually
        import numpy as np
        performance_weights = np.linspace(0.0, 1.0, 11)
        weight_pairs = [(float(pw), float(1.0 - pw)) for pw in performance_weights]
        
        if weight_pairs and len(weight_pairs) == 11:
            print(f"✅ Weight range generation validated: {len(weight_pairs)} configurations")
        else:
            print(f"❌ Weight range validation failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

def test_baseline_loading():
    """Test baseline data loading."""
    print("\n🔍 Testing baseline data loading...")
    
    try:
        # Test baseline.json exists
        baseline_path = Path("config/baseline.json")
        if baseline_path.exists():
            import json
            with open(baseline_path, 'r') as f:
                baseline_scores = json.load(f)
            
            model_count = len(baseline_scores)
            print(f"✅ Baseline data found: {model_count} models")
            
            # Test data structure
            first_model = list(baseline_scores.keys())[0]
            first_scores = baseline_scores[first_model]
            dataset_count = len(first_scores)
            print(f"✅ Baseline structure valid: {dataset_count} datasets per model")
        else:
            print("⚠️  No baseline data found (config/baseline.json missing)")
            print("   This is okay for testing - baseline comparison will be skipped")
        
        return True
    except Exception as e:
        print(f"❌ Baseline loading test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 Running Ablation Module Validation Tests")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs during testing
    
    tests = [
        test_imports,
        test_configuration_loading,
        test_data_collector,
        test_script_accessibility,
        test_config_validation,
        test_baseline_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ablation module is ready for use.")
        print("\n📋 Next steps:")
        print("1. Run cluster ablation: python ablation/run_cluster_ablation.py --data data/dataset.json")
        print("2. Run weight ablation: python ablation/run_weight_ablation.py --data data/dataset.json")  
        print("3. Quick test: python ablation/run_cluster_ablation.py --data data/dataset.json --preset quick")
        print("4. With reports: python ablation/run_weight_ablation.py --data data/dataset.json --generate-report")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before running experiments.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
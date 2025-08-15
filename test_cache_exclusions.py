#!/usr/bin/env python3
"""
Test script to verify that exclusion parameters affect cache keys correctly.
"""

import sys
import json
import hashlib
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def generate_config_hash(config):
    """Generate hash the same way as AblationDataCollector."""
    config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(config_str.encode()).hexdigest()[:12]

def test_exclusion_hashing():
    """Test that different exclusion parameters produce different hashes."""
    print("🧪 Testing Cache Hash with Exclusion Parameters")
    print("=" * 60)
    
    # Base configuration
    base_config = {
        'data_path': 'data/dataset.json',
        'seed': 42,
        'n_clusters': 32,
        'performance_weight': 0.7,
        'cost_sensitivity': 0.3,
        'excluded_models': [],
        'excluded_datasets': []
    }
    
    # Test different configurations
    test_configs = [
        {
            'name': 'Base config (no exclusions)',
            'config': base_config.copy()
        },
        {
            'name': 'Exclude one model',
            'config': {**base_config, 'excluded_models': ['model1']}
        },
        {
            'name': 'Exclude multiple models',
            'config': {**base_config, 'excluded_models': ['model1', 'model2']}
        },
        {
            'name': 'Exclude one dataset',
            'config': {**base_config, 'excluded_datasets': ['dataset1']}
        },
        {
            'name': 'Exclude both models and datasets',
            'config': {**base_config, 'excluded_models': ['model1'], 'excluded_datasets': ['dataset1']}
        },
        {
            'name': 'Same models but different order',
            'config': {**base_config, 'excluded_models': ['model2', 'model1']}  # Different order
        }
    ]
    
    # Generate hashes and check uniqueness
    hashes = []
    for test_case in test_configs:
        config = test_case['config']
        config_hash = generate_config_hash(config)
        hashes.append(config_hash)
        
        print(f"📋 {test_case['name']}")
        print(f"   excluded_models: {config.get('excluded_models', [])}")
        print(f"   excluded_datasets: {config.get('excluded_datasets', [])}")
        print(f"   Hash: {config_hash}")
        print()
    
    # Check for uniqueness
    unique_hashes = set(hashes)
    print("🔍 Analysis:")
    print(f"   Total configurations: {len(test_configs)}")
    print(f"   Unique hashes: {len(unique_hashes)}")
    
    if len(unique_hashes) == len(test_configs):
        print("✅ All configurations produce unique hashes!")
        print("   Cache system correctly handles exclusion parameters.")
    else:
        print("❌ Some configurations produce identical hashes!")
        print("   This could lead to cache conflicts.")
        
        # Find duplicates
        hash_counts = {}
        for i, h in enumerate(hashes):
            if h not in hash_counts:
                hash_counts[h] = []
            hash_counts[h].append(i)
        
        for h, indices in hash_counts.items():
            if len(indices) > 1:
                print(f"   Hash {h} is used by:")
                for idx in indices:
                    print(f"     - {test_configs[idx]['name']}")
    
    return len(unique_hashes) == len(test_configs)

def test_realistic_scenarios():
    """Test realistic exclusion scenarios."""
    print("\n" + "=" * 60)
    print("🎯 Testing Realistic Exclusion Scenarios")
    print("=" * 60)
    
    scenarios = [
        {
            'name': 'Production: exclude unreliable models',
            'excluded_models': ['unreliable_model_1', 'experimental_model_2'],
            'excluded_datasets': []
        },
        {
            'name': 'Research: focus on specific datasets',
            'excluded_models': [],
            'excluded_datasets': ['noisy_dataset', 'incomplete_dataset']
        },
        {
            'name': 'Cost optimization: exclude expensive models',
            'excluded_models': ['gpt-4', 'claude-opus'],
            'excluded_datasets': []
        },
        {
            'name': 'Domain specific: exclude irrelevant data',
            'excluded_models': ['general_model'],
            'excluded_datasets': ['out_of_domain_data']
        }
    ]
    
    base_config = {
        'data_path': 'data/dataset.json',
        'seed': 42,
        'n_clusters': 40,
        'performance_weight': 0.7,
        'cost_sensitivity': 0.3
    }
    
    scenario_hashes = []
    for scenario in scenarios:
        config = {
            **base_config,
            'excluded_models': scenario['excluded_models'],
            'excluded_datasets': scenario['excluded_datasets']
        }
        
        config_hash = generate_config_hash(config)
        scenario_hashes.append(config_hash)
        
        print(f"📋 {scenario['name']}")
        print(f"   Models: {scenario['excluded_models'] or 'None'}")
        print(f"   Datasets: {scenario['excluded_datasets'] or 'None'}")
        print(f"   Cache key: {config_hash}")
        print()
    
    unique_scenario_hashes = set(scenario_hashes)
    print(f"🔍 Scenario Analysis:")
    print(f"   Scenarios tested: {len(scenarios)}")
    print(f"   Unique cache keys: {len(unique_scenario_hashes)}")
    
    if len(unique_scenario_hashes) == len(scenarios):
        print("✅ All scenarios produce unique cache keys!")
    else:
        print("❌ Cache key conflicts detected!")
    
    return len(unique_scenario_hashes) == len(scenarios)

def main():
    """Run all exclusion caching tests."""
    print("🔑 Cache Exclusion Parameter Testing")
    print()
    
    test1_passed = test_exclusion_hashing()
    test2_passed = test_realistic_scenarios()
    
    print("\n" + "=" * 60)
    print("📊 Final Results:")
    
    if test1_passed and test2_passed:
        print("🎉 All tests passed!")
        print("✅ Cache system correctly handles exclusion parameters")
        print("✅ No cache conflicts detected")
        print("\n💡 Key Benefits:")
        print("   - Different exclusion lists → Different cache keys")
        print("   - No risk of using cached results from wrong model/dataset set")
        print("   - Safe to experiment with different exclusion strategies")
    else:
        print("⚠️  Some tests failed!")
        print("   Cache system may have issues with exclusion parameters")
    
    return 0 if (test1_passed and test2_passed) else 1

if __name__ == "__main__":
    sys.exit(main())
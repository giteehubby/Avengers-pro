"""
Simple Cluster Router

A simplified version of the cluster-based routing system that:
- Takes a single JSONL file as input 
- Uses model names directly (no mapping file needed)  
- Routes queries to best-performing models using K-means clustering
- Provides detailed performance evaluation results

Security Features:
- API keys loaded from environment variables
- Conservative token limits to prevent API errors
- Input data validation and error handling

Usage:
    export EMBEDDING_API_KEY="your-api-key"
    python simple_cluster_router.py --input data/dataset.json --output results.json
"""
import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
import os

from datasets import Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import joblib
from tqdm import tqdm
import tiktoken
from datetime import datetime

# Import local modules
from embedding_cache import EmbeddingCache
from config import SimpleClusterConfig, setup_logging


class SimpleClusterRouter:
    """
    Simple Cluster-based Model Router
    
    Routes queries to the best-performing AI models based on learned clustering patterns.
    Uses K-means clustering on query embeddings to group similar queries, then learns
    which models perform best on each cluster during training.
    
    Attributes:
        config (SimpleClusterConfig): Configuration parameters
        normalizer (Normalizer): L2 normalizer for embeddings
        embedder (EmbeddingCache): Embedding generation service
        tokenizer: Tiktoken tokenizer for precise token counting
        cluster_centers (np.ndarray): K-means cluster centers
        cluster_rankings (Dict): Model performance rankings per cluster
        kmeans_model (KMeans): Trained K-means clustering model
        available_models (List[str]): Available model names from data
    """
    
    def __init__(self, config: SimpleClusterConfig):
        """
        Initialize the cluster router.
        
        Args:
            config: Configuration object with all parameters
            
        Raises:
            ValueError: If configuration is invalid
            ConnectionError: If embedding service is unreachable
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize embedding service
        try:
            self.embedder = EmbeddingCache(
                base_url=config.embedding_base_url,
                api_key=config.embedding_api_key,
                model_name=config.embedding_model
            )
            self.logger.info(f"Initialized embedding service: {config.embedding_model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding service: {e}")
            raise ConnectionError(f"Cannot connect to embedding service: {e}")
        
        # Initialize tiktoken encoder for precise token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.embedding_model)
            self.logger.info(f"Using tiktoken encoder for {config.embedding_model}")
        except Exception:
            # Fallback to cl100k_base encoding (used by most OpenAI models)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.logger.warning(f"Fallback to cl100k_base encoding for tokenization")
        
        # Initialize model components (will be set during training)
        self.normalizer: Optional[Normalizer] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.cluster_rankings: Optional[Dict[int, Dict]] = None
        self.kmeans_model: Optional[KMeans] = None
        self.available_models: List[str] = []

    def _validate_data_item(self, item: Dict, line_num: int) -> bool:
        """
        Validate a single data item for required fields and format.
        
        Args:
            item: Data item to validate
            line_num: Line number for error reporting
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["query", "records"]
        
        for field in required_fields:
            if field not in item:
                self.logger.error(f"Line {line_num}: Missing required field '{field}'")
                return False
        
        if not isinstance(item["query"], str) or not item["query"].strip():
            self.logger.error(f"Line {line_num}: Query must be a non-empty string")
            return False
            
        if not isinstance(item["records"], dict) or not item["records"]:
            self.logger.error(f"Line {line_num}: Records must be a non-empty dict")
            return False
            
        # Validate and clean records format (model_name -> float or boolean)
        for model_name, result in item["records"].items():
            if result is None:
                # Convert None to 0.0 for missing/failed results
                item["records"][model_name] = 0.0
            elif isinstance(result, bool):
                # Convert boolean to float (True -> 1.0, False -> 0.0)
                item["records"][model_name] = 1.0 if result else 0.0
            elif isinstance(result, (int, float)):
                # Accept numeric values, convert to float
                item["records"][model_name] = float(result)
            else:
                self.logger.error(f"Line {line_num}: Record for '{model_name}' must be numeric, boolean, or null, got {type(result)}")
                return False
        
        return True

    def load_and_split_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load data from JSONL file and split by dataset with validation.
        
        Loads query-performance data from JSONL format, validates each entry,
        groups by dataset, and splits into train/test sets while maintaining
        balanced representation across datasets.
        
        Returns:
            Tuple of (train_data, test_data) as lists of dictionaries
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
            RuntimeError: If no valid data found
        """
        self.logger.info(f"Loading data from: {self.config.data_path}")
        
        # Load and validate data from JSONL file  
        dataset_data = defaultdict(list)
        data = []
        valid_items = 0
        
        try:
            with open(self.config.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        item = json.loads(line)
                        if self._validate_data_item(item, line_num):
                            data.append(item)
                            valid_items += 1
                        else:
                            self.logger.warning(f"Skipping invalid item at line {line_num}")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Line {line_num}: JSON decode error: {e}")
                        continue
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading data file: {e}")
        
        if not data:
            raise RuntimeError("No valid data found in input file")
            
        self.logger.info(f"Loaded {valid_items} valid items from {self.config.data_path}")
            
        # Extract available models from the first record
        if data and "records" in data[0]:
            all_models = list(data[0]["records"].keys())
            
            # Filter out excluded models
            self.available_models = [model for model in all_models if model not in self.config.excluded_models]
            
            # Validate that we still have models after filtering
            if not self.available_models:
                raise RuntimeError("No models available after applying exclusions. Check your excluded_models configuration.")
            
            if self.config.excluded_models:
                excluded_count = len(all_models) - len(self.available_models)
                self.logger.info(f"Excluded {excluded_count} models: {self.config.excluded_models}")
            
            self.logger.info(f"Available models ({len(self.available_models)}): {self.available_models}")
        
        # Log dataset filtering information
        if self.config.excluded_datasets:
            self.logger.info(f"Excluded datasets ({self.config.dataset_exclusion_mode} mode): {self.config.excluded_datasets}")

        # Group by dataset with validation and apply transformations
        for item in data:
            dataset_name = item.get("dataset", "default")
            
            # Skip excluded datasets only in hard mode
            if dataset_name in self.config.excluded_datasets and self.config.dataset_exclusion_mode == "hard":
                continue
            
            # Ensure all models are present in this item
            missing_models = set(self.available_models) - set(item["records"].keys())
            if missing_models:
                self.logger.warning(f"Item missing models {missing_models}, skipping")
                continue
                
            # Preserve original structure without adding default values
            data_item = {
                "query": item["query"],
                "records": item["records"],
                "dataset": dataset_name,
                "index": item.get("index", -1)
            }
            
            # Only add usages if it exists in original data
            if "usages" in item:
                data_item["usages"] = item["usages"]
            
            dataset_data[dataset_name].append(data_item)

        # Validate we have data for each dataset
        if not dataset_data:
            raise RuntimeError("No valid dataset groups found after filtering")
            
        for dataset_name, items in dataset_data.items():
            if len(items) < 2:
                self.logger.warning(f"Dataset '{dataset_name}' has only {len(items)} items, may cause split issues")

        # Split each dataset separately
        train_data = []
        test_data = []

        for dataset_name in sorted(dataset_data.keys()):
            items = dataset_data[dataset_name]
            items_sorted = sorted(items, key=lambda x: x["index"])
            
            is_excluded_dataset = dataset_name in self.config.excluded_datasets
            exclusion_mode = self.config.dataset_exclusion_mode
            
            print(f"Processing dataset {dataset_name}: {len(items_sorted)} items" + 
                  (f" (excluded - {exclusion_mode} mode)" if is_excluded_dataset else ""))
            
            # Convert to HuggingFace Dataset and split
            hf_dataset = Dataset.from_list(items_sorted)
            
            # Handle exclusion logic
            if is_excluded_dataset and exclusion_mode == "soft":
                # Soft mode: all data goes to train, nothing to test
                train_data.extend(list(hf_dataset))
                print(f"  -> Train: {len(hf_dataset)}, Test: 0 (soft excluded)")
            else:
                # Normal split for non-excluded datasets or hard mode (already filtered out)
                split_dataset = hf_dataset.train_test_split(
                    train_size=self.config.train_ratio,
                    seed=self.config.seed,
                    shuffle=True
                )
                
                train_split = split_dataset['train']
                test_split = split_dataset['test']
                
                train_data.extend(list(train_split))
                test_data.extend(list(test_split))
                
                print(f"  -> Train: {len(train_split)}, Test: {len(test_split)}")

        print(f"\nTotal - Training data: {len(train_data)}, Test data: {len(test_data)}")
        return train_data, test_data

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to fit within token limit using tiktoken for precise counting.
        
        Uses conservative token limit from config to prevent API errors.
        Preserves text integrity by encoding/decoding with tiktoken.
        
        Args:
            text: Input text to potentially truncate
            
        Returns:
            Truncated text that fits within token limits
        """
        try:
            # Encode text to tokens for precise counting
            tokens = self.tokenizer.encode(text)
            
            if len(tokens) <= self.config.max_tokens:
                return text
                
            # Truncate to max_tokens and decode back to text
            truncated_tokens = tokens[:self.config.max_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens)
            
            self.logger.info(f"Truncated text from {len(tokens)} to {len(truncated_tokens)} tokens")
            return truncated_text
            
        except Exception as e:
            self.logger.error(f"Error truncating text: {e}")
            # Fallback to character-based truncation
            max_chars = self.config.max_tokens * 3  # Conservative estimate
            return text[:max_chars] if len(text) > max_chars else text
    
    def _get_embedding_batch(self, queries_batch: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of queries."""
        embeddings = []
        for query in queries_batch:
            # Truncate query if too long
            truncated_query = self._truncate_text(query)
            embedding = np.array(self.embedder.get(truncated_query))
            if self.normalizer:
                embedding = self.normalizer.transform([embedding])[0]
            embeddings.append(embedding)
        return embeddings

    def _generate_embeddings_concurrent(self, queries: List[str]) -> List[np.ndarray]:
        """Generate embeddings for queries using concurrent processing."""
        batch_size = max(1, len(queries) // self.config.max_workers)
        query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(self._get_embedding_batch, batch) for batch in query_batches]
            
            all_embeddings = []
            with tqdm(total=len(queries), desc="Generating embeddings") as pbar:
                for future in futures:
                    batch_embeddings = future.result()
                    all_embeddings.extend(batch_embeddings)
                    pbar.update(len(batch_embeddings))
        
        return all_embeddings

    def build_cluster_model(self, train_data: List[Dict]):
        """Build cluster model and compute cluster-wise rankings from training data."""
        print(f"\n=== Building Cluster Model (n_clusters={self.config.n_clusters}) ===")
        
        queries = [item["query"] for item in train_data]
        train_records = [item["records"] for item in train_data]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(queries)} training queries...")
        train_embeddings = self._generate_embeddings_concurrent(queries)
        train_embeddings = np.array(train_embeddings)
        
        # Create and fit normalizer
        self.normalizer = Normalizer(norm='l2')
        train_embeddings = self.normalizer.fit_transform(train_embeddings)
        
        print(f"Built embeddings matrix: {train_embeddings.shape}")
        
        # Perform K-means clustering
        print(f"Performing K-means clustering with {self.config.n_clusters} clusters...")
        self.kmeans_model = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.seed,
            n_init=10
        )
        cluster_labels = self.kmeans_model.fit_predict(train_embeddings)
        self.cluster_centers = self.kmeans_model.cluster_centers_
        
        print(f"Clustering completed. Cluster centers shape: {self.cluster_centers.shape}")
        
        # Log cluster sample distribution
        cluster_counts = defaultdict(int)
        for label in cluster_labels:
            cluster_counts[label] += 1
        
        print(f"Cluster sample distribution:")
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            print(f"  Cluster {cluster_id:2d}: {count:4d} samples")
        
        # Compute cluster-wise rankings
        print("Computing cluster-wise expert rankings...")
        self.cluster_rankings = self._compute_cluster_rankings(cluster_labels, train_records)
        
        print("Cluster model built successfully")
        print(f"Clusters with data: {len(self.cluster_rankings)}")
        
        # Export cluster models if configured
        if self.config.export_cluster:
            self.export_cluster_models(self.config.export_cluster)

    def _compute_cluster_rankings(self, cluster_labels: np.ndarray, train_records: List[Dict]) -> Dict[int, Dict]:
        """Compute expert rankings for each cluster based on training performance."""
        cluster_rankings = {}
        
        # Group data by cluster
        cluster_data = defaultdict(list)
        for i, cluster_id in enumerate(cluster_labels):
            cluster_data[cluster_id].append(train_records[i])
        
        # Compute rankings for each cluster
        for cluster_id, records_list in tqdm(cluster_data.items(), desc="Computing cluster rankings"):
            if len(records_list) == 0:
                continue
                
            # Aggregate performance across all records in this cluster
            model_performance = defaultdict(list)
            
            for records in records_list:
                for model_name, score in records.items():
                    if model_name in self.available_models:
                        model_performance[model_name].append(float(score))
            
            # Calculate average performance for each model
            model_scores = {}
            for model_name, scores in model_performance.items():
                if scores:
                    model_scores[model_name] = np.mean(scores)
                else:
                    model_scores[model_name] = 0.0
            
            # Ensure all available models are included
            for model_name in self.available_models:
                if model_name not in model_scores:
                    model_scores[model_name] = 0.0
            
            # Sort models by performance (best first)
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            cluster_rankings[cluster_id] = {
                'total': len(records_list),
                'scores': dict(sorted_models),
                'ranking': [model_name for model_name, _ in sorted_models]
            }
        
        return cluster_rankings
    
    def export_cluster_models(self, export_path: str):
        """
        Export trained cluster models to disk.
        
        Args:
            export_path: Directory path to save the models
            
        Saves:
            - normalizer.joblib: L2 normalizer
            - cluster_centers.npy: K-means cluster centers
            - cluster_rankings.json: Performance rankings per cluster
        """
        if not all([self.normalizer, self.kmeans_model, self.cluster_rankings]):
            raise ValueError("Models must be trained before exporting. Call build_cluster_model() first.")
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export normalizer using joblib
            normalizer_path = export_dir / "normalizer.joblib"
            joblib.dump(self.normalizer, normalizer_path)
            self.logger.info(f"Exported normalizer to {normalizer_path}")
            
            # Export cluster centers using numpy
            centers_path = export_dir / "cluster_centers.npy" 
            np.save(centers_path, self.cluster_centers)
            self.logger.info(f"Exported cluster centers to {centers_path}")
            
            # Export cluster rankings as JSON (convert numpy int keys to strings and filter out non-serializable objects)
            rankings_serializable = {}
            for cluster_id, ranking_data in self.cluster_rankings.items():
                # Create a copy of ranking data excluding non-serializable objects
                serializable_data = {}
                for key, value in ranking_data.items():
                    if key in ['total', 'balance_scores', 'ranking', 'pareto_optimal']:
                        serializable_data[key] = value
                    # Skip 'metrics' field which contains CostEfficiencyMetrics objects
                rankings_serializable[str(cluster_id)] = serializable_data
                
            rankings_path = export_dir / "cluster_rankings.json"
            with open(rankings_path, 'w', encoding='utf-8') as f:
                json.dump(rankings_serializable, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Exported cluster rankings to {rankings_path}")
            
            # Export metadata
            metadata = {
                'n_clusters': self.config.n_clusters,
                'available_models': self.available_models,
                'embedding_model': self.config.embedding_model,
                'normalizer_type': 'l2',
                'timestamp': str(datetime.now()),
                'config': self.config.to_dict()
            }
            metadata_path = export_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Exported metadata to {metadata_path}")
            
            print(f"✅ Successfully exported cluster models to: {export_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to export cluster models: {e}")
            raise RuntimeError(f"Export failed: {e}")

    def route_queries_batch(self, queries: List[str]) -> List[List[str]]:
        """Route multiple queries in batch for better efficiency."""
        query_embeddings = self._generate_embeddings_concurrent(queries)
        query_embeddings = np.array(query_embeddings)
        query_embeddings = self.normalizer.transform(query_embeddings)
        
        # Batch distance computation
        distances = 1 - query_embeddings @ self.cluster_centers.T
        
        results = []
        for i in range(len(queries)):
            query_distances = distances[i]
            
            # Get top-k closest clusters
            closest_cluster_indices = np.argsort(query_distances)[:self.config.top_k]
            closest_distances = query_distances[closest_cluster_indices]
            
            # Convert distances to probabilities
            logits = -self.config.beta * closest_distances
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            
            # Aggregate scores across clusters
            expert_scores = defaultdict(float)
            
            for cluster_idx, prob in zip(closest_cluster_indices, probs):
                if cluster_idx not in self.cluster_rankings:
                    continue
                    
                cluster_info = self.cluster_rankings[cluster_idx]
                ranking = cluster_info['ranking']
                
                # Score based on rank position
                for model_name in self.available_models:
                    if model_name in ranking:
                        rank = ranking.index(model_name)
                        rank_score = 1.0 / (rank + 1)
                        expert_scores[model_name] += prob * rank_score
            
            # Add default scores for models not found
            for model_name in self.available_models:
                if model_name not in expert_scores:
                    expert_scores[model_name] = 0.0
            
            # Select top performing experts
            sorted_experts = sorted(expert_scores.items(), key=lambda x: x[1], reverse=True)
            selected_models = [model_name for model_name, _ in sorted_experts[:self.config.max_router]]
            results.append(selected_models)
        
        return results

    def evaluate_routing(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate routing performance on test data."""
        # Filter out excluded datasets from test data (safety check)
        filtered_test_data = []
        excluded_count = 0
        
        for item in test_data:
            dataset_name = item.get('dataset', 'default')
            if dataset_name in self.config.excluded_datasets:
                excluded_count += 1
                continue
            filtered_test_data.append(item)
        
        if excluded_count > 0:
            self.logger.warning(f"Filtered out {excluded_count} test items from excluded datasets (should have been filtered earlier)")
        
        actual_test_data = filtered_test_data
        print(f"\n=== Evaluating routing on {len(actual_test_data)} test queries ===")
        
        results = {
            'total_queries': len(actual_test_data),
            'correct_routes': 0,
            'dataset_performance': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'model_selection_stats': Counter(),
            'dataset_model_selection': defaultdict(lambda: Counter()),
            'dataset_model_accuracy': defaultdict(lambda: defaultdict(list)),
            'routing_details': []
        }
        
        # Extract all queries for batch processing
        queries = [item["query"] for item in actual_test_data]
        
        print(f"Running batch cluster routing for {len(queries)} queries...")
        
        # Use batch routing for efficiency
        with tqdm(total=len(queries), desc="Batch cluster routing") as pbar:
            batch_size = min(self.config.cluster_batch_size, len(queries))
            all_routing_results = []
            
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                batch_results = self.route_queries_batch(batch_queries)
                all_routing_results.extend(batch_results)
                pbar.update(len(batch_queries))
        
        print("Processing routing results...")
        
        # Process all results
        with tqdm(total=len(actual_test_data), desc="Evaluating results") as pbar:
            for i, test_item in enumerate(actual_test_data):
                selected_models = all_routing_results[i]
                
                routing_result = {
                    'dataset': test_item['dataset'],
                    'index': test_item.get('index', -1),
                    'selected_models': selected_models,
                    'is_correct': 0.0,
                    'true_records': {k: v for k, v in test_item['records'].items() 
                               if k in self.available_models}
                }
                
                # Check if routing is correct - use the best score from selected models
                if selected_models:
                    max_score = 0.0
                    for model_name in selected_models:
                        if model_name in test_item['records']:
                            score = test_item['records'][model_name]
                            if score > max_score:
                                max_score = score
                    routing_result['is_correct'] = max_score
                
                # Update statistics - use float score for correct routes
                results['correct_routes'] += routing_result['is_correct']
                results['dataset_performance'][routing_result['dataset']]['correct'] += routing_result['is_correct']
                
                results['dataset_performance'][routing_result['dataset']]['total'] += 1
                
                # Record model selection
                dataset = routing_result['dataset']
                for model in routing_result['selected_models']:
                    results['model_selection_stats'][model] += 1
                    results['dataset_model_selection'][dataset][model] += 1
                    results['dataset_model_accuracy'][dataset][model].append(routing_result['is_correct'])
                
                results['routing_details'].append(routing_result)
                
                current_accuracy = results['correct_routes'] / len(results['routing_details'])
                pbar.set_postfix({'accuracy': f'{current_accuracy:.4f}'})

        # Calculate accuracy as simple average across datasets (not weighted by sample count)
        dataset_accuracies = []
        for dataset, perf in results['dataset_performance'].items():
            if perf['total'] > 0:
                dataset_accuracies.append(perf['correct'] / perf['total'])
        
        results['accuracy'] = sum(dataset_accuracies) / len(dataset_accuracies) if dataset_accuracies else 0.0
        
        # Add cost analysis using filtered test data
        cost_analysis = self._analyze_routing_costs(actual_test_data, results['routing_details'], results)
        results['cost_analysis'] = cost_analysis
        
        # Add baseline comparison data for export
        baseline_scores = self._load_baseline_scores()
        if baseline_scores:
            baseline_analysis = self._analyze_baseline_comparison(baseline_scores, results, actual_test_data)
            results['baseline_analysis'] = baseline_analysis
        
        return results

    def _analyze_routing_costs(self, test_data: List[Dict], routing_details: List[Dict], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the cost implications of routing decisions.
        
        Args:
            test_data: Original test dataset with cost information
            routing_details: Detailed routing results
            results: Overall routing results for context
            
        Returns:
            Cost analysis dictionary
        """
        total_cost = 0.0
        total_queries = len(test_data)
        model_costs = defaultdict(float)
        model_queries = defaultdict(int)
        cost_per_dataset = defaultdict(float)
        correct_cost = 0.0
        incorrect_cost = 0.0
        
        for i, (test_item, routing_result) in enumerate(zip(test_data, routing_details)):
            selected_models = routing_result['selected_models']
            is_correct = routing_result['is_correct']
            dataset = routing_result['dataset']
            
            # Skip excluded datasets (safety check - should already be filtered)
            if dataset in self.config.excluded_datasets:
                continue
            
            # Get cost for selected models
            query_cost = 0.0
            usages = test_item.get('usages')
            
            for model_name in selected_models:
                cost = None
                if (usages is not None and 
                    model_name in usages and 
                    isinstance(usages[model_name], dict) and
                    'cost' in usages[model_name]):
                    cost = usages[model_name]['cost']
                
                # If no cost data available, assign zero cost
                if cost is None or not isinstance(cost, (int, float)) or cost < 0:
                    cost = 0.0  # No cost penalty for missing data
                    
                query_cost += cost
                model_costs[model_name] += cost
                model_queries[model_name] += 1
            
            total_cost += query_cost
            cost_per_dataset[dataset] += query_cost
            
            if is_correct:
                correct_cost += query_cost
            else:
                incorrect_cost += query_cost
        
        # Get accuracy from results for cost efficiency calculation
        accuracy = results.get('accuracy', 0.0)
        correct_routes = results.get('correct_routes', 0)
        
        # Calculate cost-efficiency metrics
        avg_cost_per_query = total_cost / total_queries if total_queries > 0 else 0.0
        cost_per_correct = correct_cost / correct_routes if correct_routes > 0 else 0.0
        
        # Cost efficiency: accuracy per unit cost (using simple average accuracy)
        overall_cost_efficiency = accuracy / (avg_cost_per_query + 1e-8)
        
        cost_analysis = {
            'total_cost': total_cost,
            'avg_cost_per_query': avg_cost_per_query,
            'cost_per_correct_prediction': cost_per_correct,
            'cost_efficiency': overall_cost_efficiency,
            'model_costs': dict(model_costs),
            'dataset_costs': dict(cost_per_dataset),
            'cost_distribution': {
                'correct_predictions': correct_cost,
                'incorrect_predictions': incorrect_cost
            }
        }
        
        return cost_analysis

    def _load_baseline_scores(self) -> Dict[str, Dict[str, float]]:
        """Load baseline scores from config/baseline.json if available."""
        baseline_path = Path("config/baseline.json")
        if baseline_path.exists():
            try:
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load config/baseline.json: {e}")
        return {}

    def _analyze_baseline_comparison(self, baseline_scores: Dict[str, Dict[str, float]], 
                                    results: Dict[str, Any], test_data: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze baseline comparison data for export.
        
        Args:
            baseline_scores: Baseline scores from config/baseline.json
            results: Router evaluation results
            test_data: Test data with cost information
            
        Returns:
            Baseline analysis dictionary
        """
        # Calculate per-model baseline summary
        model_summaries = []
        total_cost_by_model = {}
        
        # Calculate total costs per model from test data if available
        if test_data:
            for item in test_data:
                dataset_name = item.get('dataset', 'default')
                # Skip excluded datasets from cost calculation
                if dataset_name in self.config.excluded_datasets:
                    continue
                    
                usages = item.get('usages', {})
                for model_name, usage in usages.items():
                    if isinstance(usage, dict) and 'cost' in usage:
                        cost = usage.get('cost', 0.0)
                        if isinstance(cost, (int, float)) and cost > 0:
                            total_cost_by_model[model_name] = total_cost_by_model.get(model_name, 0.0) + cost
        
        for model, scores in baseline_scores.items():
            # Skip excluded models
            if model in self.config.excluded_models:
                continue
                
            # Calculate average score (excluding null values and excluded datasets)
            valid_scores = []
            total_datasets = 0
            for dataset, score in scores.items():
                if dataset in self.config.excluded_datasets:
                    continue
                total_datasets += 1
                if score is not None:
                    score = score / 100.0 if score > 1.0 else score
                    valid_scores.append(score)
            
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            
            # Count dataset coverage (excluding excluded datasets)
            dataset_coverage = f"{len(valid_scores)}/{total_datasets}"
            
            # Get total cost
            total_cost = total_cost_by_model.get(model, 0.0)
            
            model_summaries.append({
                'model': model,
                'avg_score': avg_score,
                'total_cost': total_cost,
                'dataset_coverage': dataset_coverage,
                'valid_datasets': len(valid_scores)
            })
        
        # Sort by average score descending
        model_summaries.sort(key=lambda x: x['avg_score'], reverse=True)
        
        # Find best overall baseline
        best_baseline = model_summaries[0] if model_summaries else None
        
        # Calculate per-dataset comparison
        dataset_comparisons = []
        for dataset, perf in results['dataset_performance'].items():
            # Skip excluded datasets (should already be filtered from results)
            if dataset in self.config.excluded_datasets:
                continue
                
            router_accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
            
            # Find best baseline model for this dataset
            dataset_baselines = {}
            for model, scores in baseline_scores.items():
                if model in self.config.excluded_models:
                    continue
                    
                score = scores.get(dataset)
                if score is not None:
                    score = score / 100.0 if score > 1.0 else score
                    dataset_baselines[model] = score
            
            if dataset_baselines:
                best_model = max(dataset_baselines.items(), key=lambda x: x[1])
                best_baseline_score = best_model[1]
                improvement = router_accuracy - best_baseline_score
                
                dataset_comparisons.append({
                    'dataset': dataset,
                    'router_accuracy': router_accuracy,
                    'best_baseline_score': best_baseline_score,
                    'best_baseline_model': best_model[0],
                    'improvement': improvement
                })
        
        return {
            'model_summaries': model_summaries,
            'best_overall_baseline': best_baseline,
            'dataset_comparisons': dataset_comparisons
        }

    def _print_baseline_summary(self, baseline_scores: Dict[str, Dict[str, float]], test_data: List[Dict] = None):
        """Print baseline model performance summary with averages and total costs."""
        print(f"\nBaseline Model Performance Summary:")
        print(f"{'Model':<35} {'Average Score':<12} {'Total Cost':<12} {'Dataset Coverage'}")
        print("-" * 77)
        
        # Calculate total costs from test data if available (excluding excluded datasets)
        model_costs = {}
        if test_data:
            model_cost_data = defaultdict(list)
            for item in test_data:
                dataset_name = item.get('dataset', 'default')
                # Skip excluded datasets from cost calculation
                if dataset_name in self.config.excluded_datasets:
                    continue
                    
                usages = item.get('usages')
                if usages:
                    for model_name, usage in usages.items():
                        if isinstance(usage, dict) and 'cost' in usage:
                            cost = usage['cost']
                            if isinstance(cost, (int, float)) and cost >= 0:
                                model_cost_data[model_name].append(cost)
            
            # Calculate total costs
            for model_name, costs in model_cost_data.items():
                if costs:
                    model_costs[model_name] = sum(costs)

        model_averages = []
        for model, scores in baseline_scores.items():
            # Skip excluded models
            if model in self.config.excluded_models:
                continue
                
            # Calculate average excluding null values and excluded datasets
            valid_scores = []
            total_datasets = 0
            for dataset, score in scores.items():
                if dataset in self.config.excluded_datasets:
                    continue
                total_datasets += 1
                if score is not None:
                    score = score / 100.0 if score > 1.0 else score
                    valid_scores.append(score)
            
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                coverage = f"{len(valid_scores)}/{total_datasets}"
                total_cost = model_costs.get(model, 0.0)
                model_averages.append((model, avg_score, total_cost, coverage))
        
        # Sort by average score (descending)
        model_averages.sort(key=lambda x: x[1], reverse=True)
        
        for model, avg_score, total_cost, coverage in model_averages:
            model_short = model.split('/')[-1]  # Show only model name without provider
            cost_str = f"${total_cost:.4f}" if total_cost > 0 else "N/A"
            print(f"{model_short:<35} {avg_score:.4f}        {cost_str:<12} {coverage}")
        
        if model_averages:
            best_model = model_averages[0]
            best_total_cost = best_model[2]
            cost_info = f", total cost: ${best_total_cost:.4f}" if best_total_cost > 0 else ""
            print(f"\nBest Overall Baseline: {best_model[0].split('/')[-1]} (avg: {best_model[1]:.4f}{cost_info})")

    def print_evaluation_results(self, results: Dict[str, Any], test_data: List[Dict] = None):
        """Print detailed evaluation results with baseline comparison."""
        print(f"\n{'='*50}")
        print("SIMPLE CLUSTER ROUTING EVALUATION RESULTS")
        print(f"{'='*50}")
        
        # Display excluded models/datasets if any
        if self.config.excluded_models:
            print(f"Excluded Models: {', '.join(self.config.excluded_models)}")
        if self.config.excluded_datasets:
            print(f"Excluded Datasets: {', '.join(self.config.excluded_datasets)}")
        if self.config.excluded_models or self.config.excluded_datasets:
            print()
        
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        
        # Load baseline scores for comparison
        baseline_scores = self._load_baseline_scores()
        
        # Print baseline model averages if available
        if baseline_scores:
            self._print_baseline_summary(baseline_scores, test_data)
        
        print(f"\nPer-Dataset Performance:")
        if baseline_scores:
            print(f"{'Dataset':<15} {'Router':<8} {'Baseline':<9} {'Best Model':<12} {'Improvement':<12}")
            print("-" * 70)
        
        for dataset, perf in results['dataset_performance'].items():
            router_accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
            
            if baseline_scores:
                # Find best baseline model for this dataset (excluding excluded models)
                dataset_baselines = {}
                for model, scores in baseline_scores.items():
                    # Skip excluded models
                    if model in self.config.excluded_models:
                        continue
                        
                    score = scores.get(dataset)
                    if score is not None:
                        # Convert percentage to decimal if needed
                        score = score / 100.0 if score > 1.0 else score
                        dataset_baselines[model] = score
                    else:
                        dataset_baselines[model] = 0.0
                
                if dataset_baselines:
                    best_model = max(dataset_baselines.items(), key=lambda x: x[1])
                    best_baseline = best_model[1]
                    improvement = router_accuracy - best_baseline
                    improvement_str = f"{improvement:+.4f}" if improvement != 0 else "0.000"
                    
                    print(f"{dataset:<15} {router_accuracy:.4f}    {best_baseline:.4f}     {best_model[0].split('/')[-1]:<12} {improvement_str}")
                else:
                    print(f"  {dataset:15}: {router_accuracy:.4f} ({perf['correct']}/{perf['total']})")
            else:
                print(f"  {dataset:15}: {router_accuracy:.4f} ({perf['correct']}/{perf['total']})")
        
        print(f"\nModel Selection Frequency:")
        total_selections = sum(results['model_selection_stats'].values())
        for model, count in results['model_selection_stats'].most_common():
            percentage = count / total_selections * 100 if total_selections > 0 else 0
            print(f"  {model:30}: {count:4d} ({percentage:5.1f}%)")
        
        # Print per-dataset model selection and performance
        if 'dataset_model_selection' in results and results['dataset_model_selection']:
            print(f"\nPer-Dataset Model Selection & Performance:")
            print(f"Dataset         Model                    Selection   Accuracy")
            print(f"-" * 58)
            
            for dataset in sorted(results['dataset_model_selection'].keys()):
                dataset_selections = results['dataset_model_selection'][dataset]
                dataset_total = sum(dataset_selections.values())
                
                first_model = True
                for model, count in dataset_selections.most_common():
                    percentage = count / dataset_total * 100 if dataset_total > 0 else 0
                    
                    # Calculate accuracy for this model on this dataset
                    accuracy_scores = results['dataset_model_accuracy'][dataset][model]
                    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
                    
                    dataset_label = dataset if first_model else ""
                    print(f"{dataset_label:<15} {model:<24} {count:3d} ({percentage:4.1f}%)  {avg_accuracy:.4f}")
                    first_model = False
        
        # Print overall model statistics
        if 'model_selection_stats' in results and results['model_selection_stats']:
            print(f"\nOverall Model Statistics:")
            print(f"Model                           Total Selection   Selection Rate   Avg Accuracy")
            print(f"-" * 77)
            
            for model, count in results['model_selection_stats'].most_common():
                percentage = count / total_selections * 100 if total_selections > 0 else 0
                
                # Calculate overall accuracy for this model across all datasets
                all_scores = []
                for dataset in results['dataset_model_accuracy']:
                    if model in results['dataset_model_accuracy'][dataset]:
                        all_scores.extend(results['dataset_model_accuracy'][dataset][model])
                
                avg_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0
                print(f"{model:<31} {count:<13} {percentage:5.1f}%          {avg_accuracy:.4f}")
        
        # Print cost analysis
        if 'cost_analysis' in results:
            cost_analysis = results['cost_analysis']
            
            print(f"\n{'='*50}")
            print("COST-EFFICIENCY ANALYSIS")
            print(f"{'='*50}")
            
            print(f"Total Cost: ${cost_analysis['total_cost']:.4f}")
            print(f"Average Cost per Query: ${cost_analysis['avg_cost_per_query']:.4f}")
            print(f"Cost per Correct Prediction: ${cost_analysis['cost_per_correct_prediction']:.4f}")
            print(f"Cost Efficiency (Accuracy/Cost): {cost_analysis['cost_efficiency']:.4f}")
            
            print(f"\nCost by Dataset:")
            for dataset, cost in cost_analysis['dataset_costs'].items():
                avg_cost = cost / results['dataset_performance'][dataset]['total']
                print(f"  {dataset:15}: ${cost:.4f} (${avg_cost:.4f} per query)")
            
            print(f"\nModel Cost Distribution:")
            total_model_cost = sum(cost_analysis['model_costs'].values())
            for model, cost in sorted(cost_analysis['model_costs'].items(), 
                                    key=lambda x: x[1], reverse=True):
                percentage = (cost / total_model_cost) * 100 if total_model_cost > 0 else 0
                print(f"  {model:30}: ${cost:8.4f} ({percentage:5.1f}%)")

    def run_routing(self):
        """Run the complete cluster routing process."""
        print("Starting Simple Cluster Router")
        print(f"Configuration: n_clusters={self.config.n_clusters}, max_router={self.config.max_router}, train_ratio={self.config.train_ratio}")
        
        # Load and split data
        train_data, test_data = self.load_and_split_data()
        
        # Build cluster model
        self.build_cluster_model(train_data)
        
        # Evaluate routing
        results = self.evaluate_routing(test_data)
        
        # Save results if path specified in config
        if self.config.results_path:
            import os
            os.makedirs(os.path.dirname(self.config.results_path), exist_ok=True)
            with open(self.config.results_path, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print results
        self.print_evaluation_results(results, test_data)
        
        return results


def main():
    """
    Main entry point for Simple Cluster Router.
    
    Parses command line arguments, initializes configuration, and runs the routing evaluation.
    Supports loading configuration from environment variables and command line overrides.
    """
    parser = argparse.ArgumentParser(
        description='Simple Cluster Router - AI Model Routing via K-means Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with environment variables
    export EMBEDDING_API_KEY="your-key"
    python simple_cluster_router.py --input data/dataset.json --output results.json

    # Override default parameters  
    python simple_cluster_router.py --input data.json --clusters 64 --train_ratio 0.8

    # Use custom results path
    python simple_cluster_router.py --input data.json --results_path results/custom_results.json

    # Use configuration file
    python simple_cluster_router.py --config config.json --input data.json
        """
    )
    
    parser.add_argument('--input', type=str, required=True, 
                       help='Input JSONL file path with query-model performance data')
    parser.add_argument('--output', type=str, 
                       help='Output file path for results (JSON format)')
    parser.add_argument('--config', type=str, 
                       help='Configuration file path (JSON format)')
    parser.add_argument('--clusters', type=int, default=32, 
                       help='Number of K-means clusters (default: 32)')
    parser.add_argument('--train_ratio', type=float, default=0.7, 
                       help='Training data ratio (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--max_router', type=int, default=1, 
                       help='Number of top models to select (default: 1)')
    parser.add_argument('--max_tokens', type=int, default=7500,
                       help='Maximum tokens per query (default: 7500)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--excluded_models', type=str,
                       help='Comma-separated list of models to exclude from routing')
    parser.add_argument('--excluded_datasets', type=str,
                       help='Comma-separated list of datasets to exclude from evaluation')
    parser.add_argument('--dataset_exclusion_mode', choices=['soft', 'hard'], default='hard',
                       help='Dataset exclusion mode: soft (exclude from eval only) or hard (exclude completely)')
    parser.add_argument('--export_cluster', type=str,
                       help='Directory path to export trained cluster models (normalizer, centers, rankings)')
    parser.add_argument('--results_path', type=str,
                       help='Path to save routing results JSON file (alternative to --output)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config = SimpleClusterConfig.from_file(args.config, args.input)
        else:
            logger.info("Loading configuration from environment and arguments")
            # Parse excluded models and datasets from command line
            excluded_models = []
            if args.excluded_models:
                excluded_models = [model.strip() for model in args.excluded_models.split(",") if model.strip()]
            
            excluded_datasets = []
            if args.excluded_datasets:
                excluded_datasets = [dataset.strip() for dataset in args.excluded_datasets.split(",") if dataset.strip()]
            
            config = SimpleClusterConfig.from_env(
                data_path=args.input,
                n_clusters=args.clusters,
                train_ratio=args.train_ratio,
                seed=args.seed,
                max_router=args.max_router,
                max_tokens=args.max_tokens,
                excluded_models=excluded_models,
                excluded_datasets=excluded_datasets,
                dataset_exclusion_mode=args.dataset_exclusion_mode,
                export_cluster=args.export_cluster,
                results_path=args.results_path
            )
        
        logger.info(f"Configuration: {config.to_dict()}")
        
        # Initialize and run router
        logger.info("Initializing Simple Cluster Router")
        router = SimpleClusterRouter(config)
        
        logger.info("Starting routing evaluation")
        results = router.run_routing()
        
        # Save results if output path specified
        if args.output:
            from datetime import datetime
            results_serializable = {
                'timestamp': datetime.now().isoformat(),
                'config': config.to_dict(),
                'results': {
                    'accuracy': results['accuracy'],
                    'correct_routes': results['correct_routes'],
                    'total_queries': results['total_queries'],
                    'dataset_performance': dict(results['dataset_performance']),
                    'model_selection_stats': dict(results['model_selection_stats'])
                }
            }
            
            # Ensure output is in results directory
            output_path = Path("results") / Path(args.output).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_serializable, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_path}")
            print(f"\nResults saved to: {output_path}")
        
        logger.info("Routing evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
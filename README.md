# Avengers-Pro: Test-Time Routing Framework

Balancing performance and efficiency is a central challenge in large language model (LLM) advancement. GPT-5 addresses this with test-time routing, dynamically assigning queries to either an efficient or a high-capacity model. In this work, we present **Avengers-Pro**, a test-time routing framework that ensembles LLMs of varying capacities and efficiencies.

The **Avengers-Pro** embeds and clusters incoming queries, then routes each to the most suitable model based on a performance-efficiency score. Across 6 challenging benchmarks and 8 leading models---including GPT-5-medium, Gemini-2.5-pro, and Claude-opus-4.1---the **Avengers-Pro** achieves state-of-the-art results: by varying a performance-efficiency parameter, it can **surpass the strongest single model** (GPT-5-medium) by **+8% in average accuracy**. Moreover, it can **match** the average accuracy of the strongest single model at **−42% lower cost**, and reach ~**90%** of that performance at **−74% lower cost**. Last but not least, it achieves Pareto-optimality, consistently yielding the highest accuracy for any given cost, and the lowest cost for any given accuracy, among all single models.

## 🚀 Features

- **Intelligent Routing**: Model selection based on query semantic similarity
- **K-means Clustering**: Groups similar queries to learn optimal model patterns
- **Performance-Efficiency Trade-off**: Balances accuracy and computational cost
- **Multi-Model Support**: Works with 8+ leading LLMs
- **Pareto-Optimal**: Best accuracy-cost trade-offs across all scenarios

## 📋 Requirements

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## 🛠️ Setup

1. **Environment Variables**
   ```bash
   export EMBEDDING_API_KEY="your-embedding-api-key"
   export EMBEDDING_BASE_URL="http://your-embedding-service:port/v1"
   ```

2. **Data Format**
   
   Input file in JSONL format:
   ```json
   {
     "query": "Your query text...",
     "records": {
       "anthropic/claude-opus-4.1": true,
       "google/gemini-2.5-pro": false,
       "openai/gpt-5-chat": true
     },
     "dataset": "arc-agi-v1",
     "index": 0
   }
   ```

## 🔧 Usage

### Basic Usage

```bash
# Set API key
export EMBEDDING_API_KEY="your-api-key"

# Run routing evaluation
python simple_cluster_router.py --input data/dataset.json --output results.json
```

### Advanced Configuration

```bash
# Custom clustering parameters
python simple_cluster_router.py \
  --input data/dataset.json \
  --output results.json \
  --clusters 64 \
  --train_ratio 0.8 \
  --max_router 2 \
  --max_tokens 7000
```

## 📊 Core Algorithm

1. **Training Phase**: Load data → Generate embeddings → K-means clustering → Calculate model rankings
2. **Routing Phase**: New query embedding → Find nearest clusters → Aggregate scores → Select optimal model

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clusters` | 32 | Number of K-means clusters |
| `train_ratio` | 0.7 | Training data ratio |
| `beta` | 9.0 | Temperature parameter for cluster selection |
| `max_tokens` | 7500 | Maximum tokens per query |

## 📈 Results

The framework consistently achieves:
- **+8% accuracy** over strongest single model
- **42% cost reduction** while maintaining accuracy
- **74% cost reduction** at 90% performance
- **Pareto-optimal** performance across all metrics

## 📁 Project Structure

```
cluster/
├── README.md
├── simple_cluster_router.py     # Main router
├── balance_cluster_router.py    # Balance-aware router  
├── config.py                   # Configuration management
├── embedding_cache.py          # Embedding cache
├── data/                       # Input data
└── results/                   # Output results
```

## 🤝 Contributing

1. Fork the project
2. Create a feature branch
3. Submit changes
4. Create a Pull Request

## 📄 License

MIT License

---

*For detailed technical implementation and experimental results, please refer to our paper.*
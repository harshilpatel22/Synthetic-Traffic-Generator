# Synthetic Network Traffic Data Generator Course Project For Network And Information Security BITE401L


A robust tool for generating synthetic network traffic data that closely mimics real traffic patterns, preserving statistical properties while enabling better anomaly detection model training.

## Overview

This project provides an advanced synthetic data generator specifically designed for network traffic analysis and anomaly detection. The tool analyzes real network traffic data, learns its statistical properties and correlations, and generates realistic synthetic data that can be used for:

- Training machine learning models when real data is limited
- Augmenting datasets for better anomaly detection
- Testing and benchmarking without exposing sensitive network data
- Research and development of new detection techniques

## Features

- **Statistical Property Preservation**: Maintains distributions and correlations between features
- **Correlation-Aware Generation**: Respects the relationships between traffic features
- **Anomaly Synthesis**: Generates both normal and anomalous traffic patterns
- **Clustering-Based Generation**: Identifies and preserves distinct traffic patterns
- **Quality Evaluation**: Built-in methods to evaluate synthetic data quality
- **Visualization Tools**: Compare real vs synthetic data distributions

## Project Structure

```
├── synthetic_data_generator.py   # Main class for synthetic data generation
├── process.py                    # Data preprocessing script for merging datasets
├── neweval.py                    # Evaluation script for testing generator
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synthetic-network-traffic-generator.git
cd synthetic-network-traffic-generator

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy
- xgboost
- tensorflow

## Usage

### Basic Usage

```python
from synthetic_data_generator import ImprovedSyntheticDataGenerator

# Initialize the generator
generator = ImprovedSyntheticDataGenerator(
    real_data_path="your_data.csv",
    features=["feature1", "feature2", "feature3"],
    label_column="Label",
    anomaly_label=1
)

# Load and preprocess the data
real_data = generator.load_data()

# Generate synthetic data (twice the size of real data)
synthetic_data = generator.generate_synthetic_data(
    n_samples=len(real_data) * 2,
    normal_ratio=0.9  # 90% normal traffic, 10% anomalous
)

# Visualize comparison between real and synthetic data
generator.visualize_comparison()

# Evaluate synthetic data quality using ML models
evaluation_results = generator.evaluate_with_ml_models(test_size=0.2)
```

### Data Preprocessing

The `process.py` script demonstrates how to preprocess and merge multiple traffic datasets:

```bash
python process.py
```

This script:
1. Loads traffic datasets (CICIDS and Darknet)
2. Standardizes column names
3. Extracts common features
4. Normalizes the data
5. Combines datasets for more comprehensive training

### Evaluation

The `neweval.py` script shows how to:
1. Label traffic data using Isolation Forest for anomaly detection
2. Generate synthetic data based on the labeled dataset
3. Evaluate the quality of synthetic data by training ML models

```bash
python neweval.py
```

## How It Works

The generator uses a multi-step approach to create high-quality synthetic data:

1. **Data Analysis**: Analyzes feature distributions and correlations in real data
2. **Clustering**: Identifies distinct traffic patterns using K-means clustering
3. **Synthetic Generation**: Creates new samples using a mix of:
   - Cluster-based generation (70%)
   - Direct sampling with noise (30%)
4. **Correlation Preservation**: Applies corrections to maintain feature relationships
5. **Anomaly Generation**: Creates anomalous samples by either:
   - Sampling from existing anomalies with amplification
   - Modifying normal samples to create new anomaly patterns
6. **Validation**: Provides tools to validate synthetic data quality

## Key Components

### ImprovedSyntheticDataGenerator Class

The main class that handles:
- Loading and preprocessing real data
- Analyzing feature distributions and correlations
- Clustering data to identify patterns
- Generating synthetic samples
- Visualizing and evaluating results

### Feature Distribution Analysis

For each feature, the generator analyzes:
- Basic statistics (min, max, mean, median, std)
- Distribution parameters
- Outlier thresholds
- Skewness and kurtosis

### Synthetic Data Generation

The generation process:
1. Scales the data for better modeling
2. Separates normal and anomalous patterns
3. Applies clustering to identify traffic patterns
4. Generates new samples with controlled variance
5. Applies correlation-aware corrections
6. Rescales back to original feature ranges

## Evaluation

The synthetic data is evaluated using multiple approaches:

1. **PCA Visualization**: Compares real and synthetic data in reduced dimensions
2. **Correlation Matrix Comparison**: Ensures feature relationships are preserved
3. **Distribution Comparison**: Validates feature distributions match
4. **ML Model Performance**: Trains models on synthetic data and tests on real data
5. You can find result of the project in result.png file

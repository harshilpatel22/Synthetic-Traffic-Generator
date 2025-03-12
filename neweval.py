import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report
import traceback
import sys

# Import the improved generator
from synthetic_data_generator import ImprovedSyntheticDataGenerator

# Define the features and load the data
features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Bwd Packet Length Max']

# Load data
print("Loading dataset...")
df = pd.read_csv("merged_encrypted_traffic.csv")
X = df[features]

# Detect anomalies
print("Detecting anomalies with Isolation Forest...")
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['Label'] = iso_forest.fit_predict(X)
# Convert to binary (1 for anomalies, 0 for normal)
df['Label'] = df['Label'].map({1: 0, -1: 1})

# Save labeled dataset
print("Saving labeled dataset...")
df.to_csv("labeled_encrypted_traffic.csv", index=False)

label_column = "Label"
anomaly_label = 1

# Initialize the generator
print("Initializing generator...")
generator = ImprovedSyntheticDataGenerator(
    real_data_path="labeled_encrypted_traffic.csv",
    features=features,
    label_column=label_column,
    anomaly_label=anomaly_label
)

# Load and preprocess the data
print("Loading and preprocessing data...")
real_data = generator.load_data()
print(f"Loaded real data with shape: {real_data.shape}")
print(f"Real data columns: {real_data.columns.tolist()}")
print(f"Is label column in real data? {label_column in real_data.columns}")

# Generate synthetic data
print("Generating synthetic data...")
synthetic_data = generator.generate_synthetic_data(
    n_samples=len(real_data) * 2,
    normal_ratio=0.9
)
print(f"Generated synthetic data with shape: {synthetic_data.shape}")
print(f"Synthetic data columns: {synthetic_data.columns.tolist()}")

# Save the synthetic data
print("Saving synthetic data...")
synthetic_data.to_csv("improved_synthetic_traffic.csv", index=False)
print("Saved synthetic data to 'improved_synthetic_traffic.csv'")

# Skip visualization to isolate the problem
print("Skipping visualization for troubleshooting...")
# generator.visualize_comparison(methods=['pca', 'correlation', 'distribution'])

# Evaluate with clear error handling
print("\nChecking if label column exists:", label_column in real_data.columns)
if label_column in real_data.columns:
    print("Starting evaluation with ML models...")
    try:
        evaluation_results = generator.evaluate_with_ml_models(test_size=0.2)
        
        # Print summary
        print("\nSynthetic Data Quality Summary:")
        print(f"Model accuracy when trained on real data: {evaluation_results['real_to_real']['accuracy']:.4f}")
        print(f"Model accuracy when trained on synthetic data: {evaluation_results['synth_to_real']['accuracy']:.4f}")
        
        # Calculate relative performance
        relative_performance = evaluation_results['synth_to_real']['accuracy'] / evaluation_results['real_to_real']['accuracy'] * 100
        print(f"Relative performance: {relative_performance:.2f}%")
        
        if relative_performance >= 90:
            print("EXCELLENT: Synthetic data performs very similarly to real data")
        elif relative_performance >= 80:
            print("GOOD: Synthetic data captures most patterns in the real data")
        elif relative_performance >= 70:
            print("ACCEPTABLE: Synthetic data is useful but misses some patterns")
        else:
            print("NEEDS IMPROVEMENT: Synthetic data is not adequately representative")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Traceback:")
        traceback.print_exc(file=sys.stdout)
else:
    print(f"Evaluation skipped: Label column '{label_column}' not found in real data columns.")
    print(f"Available columns: {real_data.columns.tolist()}")
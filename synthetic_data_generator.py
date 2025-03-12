import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ImprovedSyntheticDataGenerator:
    def __init__(self, real_data_path, features=None, label_column=None, anomaly_label=None):
        """
        Initialize the generator with dataset configuration
        
        Parameters:
        -----------
        real_data_path: str
            Path to the real dataset CSV file
        features: list, optional
            List of feature columns to use
        label_column: str, optional
            Name of the label column if available
        anomaly_label: any, optional
            Value in label column that indicates anomalous traffic
        """
        self.real_data_path = real_data_path
        self.features = features
        self.label_column = label_column
        self.anomaly_label = anomaly_label
        self.real_data = None
        self.synthetic_data = None
        self.scaler = None
        self.feature_distributions = {}
        self.correlation_matrix = None
        
    def load_data(self):
        """Load and preprocess the real data"""
        # Load data
        self.real_data = pd.read_csv(self.real_data_path)
        
        # If features specified, select only those columns
        if self.features:
            if self.label_column and self.label_column not in self.features:
                self.real_data = self.real_data[self.features + [self.label_column]]
            else:
                self.real_data = self.real_data[self.features]
        
        # Handle missing values
        self.real_data.fillna(self.real_data.median(), inplace=True)
        
        # Store information about feature distributions and correlations
        self._analyze_feature_distributions()
        
        return self.real_data
    
    def _analyze_feature_distributions(self):
        """Analyze the statistical properties of each feature"""
        # Select numerical features only
        data_for_analysis = self.real_data.drop(columns=[self.label_column]) if self.label_column else self.real_data
        
        # Compute correlation matrix
        self.correlation_matrix = data_for_analysis.corr()
        
        # Analyze each feature's distribution
        for column in data_for_analysis.columns:
            data = data_for_analysis[column].values
            
            # Fit a distribution to the data
            # Try normal, lognormal and exponential distributions
            distributions = []
            try:
                # Normal distribution
                norm_params = stats.norm.fit(data)
                distributions.append(('normal', norm_params))
                
                # For positive data, try lognormal
                if data.min() >= 0:
                    lognorm_params = stats.lognorm.fit(data + 1e-10)  # Add small value to avoid zeros
                    distributions.append(('lognormal', lognorm_params))
                    
                    # Try exponential for decay patterns
                    exp_params = stats.expon.fit(data)
                    distributions.append(('exponential', exp_params))
            except:
                pass
            
            # Store basic statistics and best-fit distribution
            self.feature_distributions[column] = {
                'min': data.min(),
                'max': data.max(),
                'mean': data.mean(),
                'median': np.median(data),
                'std': data.std(),
                'skew': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'q1': np.percentile(data, 25),
                'q3': np.percentile(data, 75),
                'distributions': distributions,
                'outlier_threshold_lower': np.percentile(data, 0.5),
                'outlier_threshold_upper': np.percentile(data, 99.5)
            }
    
    def preprocess_for_synthesis(self):
        """Prepare data for synthesis by scaling and separating normal/anomalous data"""
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Select numerical features
        if self.label_column:
            X = self.real_data.drop(columns=[self.label_column])
            y = self.real_data[self.label_column]
        else:
            X = self.real_data
            y = None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Separate normal and anomalous data if labels are provided
        if self.label_column and self.anomaly_label is not None:
            normal_indices = y != self.anomaly_label
            anomalous_indices = y == self.anomaly_label
            
            X_normal = X_scaled[normal_indices]
            X_anomalous = X_scaled[anomalous_indices]
            
            return X_scaled, X_normal, X_anomalous, X.columns
        else:
            # Use Isolation Forest to identify potential anomalies
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(X_scaled)
            
            normal_indices = predictions == 1
            anomalous_indices = predictions == -1
            
            X_normal = X_scaled[normal_indices]
            X_anomalous = X_scaled[anomalous_indices]
            
            return X_scaled, X_normal, X_anomalous, X.columns
    
    def cluster_data(self, X_scaled, n_clusters=5):
        """Cluster the data to identify different traffic patterns"""
        # Fit K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Count instances per cluster
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_props = dict(zip(unique, counts / len(clusters)))
        
        return clusters, cluster_centers, cluster_props
    
    def generate_synthetic_data(self, n_samples, normal_ratio=0.9):
        """
        Generate synthetic data that mimics the real data distribution
        
        Parameters:
        -----------
        n_samples: int
            Number of synthetic samples to generate
        normal_ratio: float
            Proportion of normal (non-anomalous) data to generate
        
        Returns:
        --------
        pd.DataFrame
            DataFrame of synthetic data
        """
        # Preprocess the real data
        X_scaled, X_normal, X_anomalous, feature_names = self.preprocess_for_synthesis()
        
        # Cluster normal and anomalous data separately
        normal_clusters, normal_centers, normal_props = self.cluster_data(X_normal, n_clusters=min(5, len(X_normal) // 100))
        
        if len(X_anomalous) > 0:
            anomaly_clusters, anomaly_centers, anomaly_props = self.cluster_data(
                X_anomalous, 
                n_clusters=min(3, max(1, len(X_anomalous) // 50))
            )
        
        # Generate synthetic data
        n_normal = int(n_samples * normal_ratio)
        n_anomalous = n_samples - n_normal
        
        # Initialize synthetic data arrays
        synthetic_normal = np.zeros((n_normal, X_scaled.shape[1]))
        synthetic_anomalous = np.zeros((n_anomalous, X_scaled.shape[1]))
        
        # Generate normal samples using a mixture of methods
        for i in range(n_normal):
            # With 70% probability, use cluster-based generation
            if np.random.random() < 0.7:
                # Select a random cluster based on its proportion
                cluster_idx = np.random.choice(list(normal_props.keys()), p=list(normal_props.values()))
                
                # Get cluster center
                center = normal_centers[cluster_idx]
                
                # Generate sample around the cluster center with controlled variance
                sample = center + np.random.normal(0, 0.5, size=center.shape)
            else:
                # With 30% probability, use direct sampling from the normal data
                idx = np.random.randint(0, len(X_normal))
                base_sample = X_normal[idx]
                
                # Add small random noise to avoid exact duplication
                sample = base_sample + np.random.normal(0, 0.2, size=base_sample.shape)
            
            # Apply correlation-aware corrections
            for j in range(len(sample)):
                for k in range(j+1, len(sample)):
                    if abs(self.correlation_matrix.iloc[j, k]) > 0.5:
                        correlation = self.correlation_matrix.iloc[j, k]
                        # Adjust values to maintain correlation
                        sample[k] = correlation * sample[j] + np.random.normal(0, 0.1) * (1 - abs(correlation))
            
            synthetic_normal[i] = sample
        
        # Generate anomalous samples
        if n_anomalous > 0:
            if len(X_anomalous) > 0:
                # If we have anomalous examples, use them for generation
                for i in range(n_anomalous):
                    if np.random.random() < 0.6 and len(anomaly_centers) > 0:
                        # Select a random anomaly cluster
                        cluster_idx = np.random.choice(list(anomaly_props.keys()), p=list(anomaly_props.values()))
                        center = anomaly_centers[cluster_idx]
                        
                        # Generate more extreme variations of anomalies
                        sample = center + np.random.normal(0, 0.8, size=center.shape)
                    else:
                        # Sample from existing anomalies with amplification
                        idx = np.random.randint(0, len(X_anomalous))
                        base_sample = X_anomalous[idx]
                        
                        # Amplify the anomalous characteristics
                        amplification = 1.2 + np.random.random() * 0.5  # 1.2 to 1.7x
                        sample = base_sample * amplification
                    
                    synthetic_anomalous[i] = sample
            else:
                # If no anomalous examples, generate synthetic anomalies
                # by exaggerating the patterns in normal data
                for i in range(n_anomalous):
                    # Pick a normal sample and modify it to be an anomaly
                    idx = np.random.randint(0, len(X_normal))
                    base_sample = X_normal[idx]
                    
                    # Modify 2-4 random features significantly
                    n_features_to_modify = np.random.randint(2, min(5, X_scaled.shape[1]))
                    features_to_modify = np.random.choice(range(X_scaled.shape[1]), size=n_features_to_modify, replace=False)
                    
                    # Create anomalous sample
                    sample = base_sample.copy()
                    for feat_idx in features_to_modify:
                        # Either make the value extremely large or extremely small
                        if np.random.random() < 0.5:
                            sample[feat_idx] = base_sample[feat_idx] * (3 + np.random.random() * 2)
                        else:
                            sample[feat_idx] = base_sample[feat_idx] * (0.1 + np.random.random() * 0.2)
                    
                    synthetic_anomalous[i] = sample
        
        # Combine normal and anomalous samples
        synthetic_combined = np.vstack([synthetic_normal, synthetic_anomalous])
        
        # Create labels if needed
        if self.label_column:
            y_synthetic = np.zeros(n_samples)
            if self.anomaly_label is not None:
                y_synthetic[n_normal:] = self.anomaly_label
        
        # Inverse transform to get the original scale
        synthetic_data_orig_scale = self.scaler.inverse_transform(synthetic_combined)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_data_orig_scale, columns=feature_names)
        
        # Apply constraints to ensure data validity
        for column in synthetic_df.columns:
            stats = self.feature_distributions[column]
            
            # Ensure values are within plausible ranges (with a bit of controlled extrapolation)
            min_val = stats['min'] - 0.1 * abs(stats['min'])
            max_val = stats['max'] + 0.1 * abs(stats['max'])
            
            # For columns that must be non-negative
            if stats['min'] >= 0:
                min_val = max(0, min_val)
                synthetic_df[column] = synthetic_df[column].clip(lower=0)
            
            # Apply general clipping with a slight allowance for new patterns
            synthetic_df[column] = synthetic_df[column].clip(lower=min_val, upper=max_val)
            
            # For integer features, round values
            if np.issubdtype(self.real_data[column].dtype, np.integer):
                synthetic_df[column] = np.round(synthetic_df[column]).astype(int)
        
        # Add label column if it exists
        if self.label_column and self.anomaly_label is not None:
            synthetic_df[self.label_column] = y_synthetic
        
        self.synthetic_data = synthetic_df
        return synthetic_df
    
    def visualize_comparison(self, methods=['pca', 'correlation', 'distribution']):
        """Visualize comparison between real and synthetic data"""
        if self.real_data is None or self.synthetic_data is None:
            print("Both real and synthetic data must be loaded first")
            return
        
        # Filter out the label column for analysis
        real_X = self.real_data.drop(columns=[self.label_column]) if self.label_column else self.real_data
        synth_X = self.synthetic_data.drop(columns=[self.label_column]) if self.label_column in self.synthetic_data.columns else self.synthetic_data
        
        # Standardize data for visualization
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_X)
        synth_scaled = scaler.transform(synth_X)
        
        plt.figure(figsize=(15, 10))
        
        if 'pca' in methods:
            # PCA visualization
            pca = PCA(n_components=2)
            real_pca = pca.fit_transform(real_scaled)
            synth_pca = pca.transform(synth_scaled)
            
            plt.subplot(2, 2, 1)
            plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label="Real", s=10, color='blue')
            plt.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, label="Synthetic", s=10, color='red')
            plt.title("PCA Projection")
            plt.legend()
            plt.grid(True)
        
        if 'correlation' in methods:
            # Correlation matrix comparison
            plt.subplot(2, 2, 2)
            real_corr = real_X.corr()
            synth_corr = synth_X.corr()
            
            # Calculate correlation difference
            corr_diff = np.abs(real_corr - synth_corr)
            
            sns.heatmap(corr_diff, annot=True, cmap='Blues', vmin=0, vmax=1)
            plt.title("Correlation Matrix Difference (lower is better)")
        
        if 'distribution' in methods:
            # Distribution comparison for key features
            plt.subplot(2, 2, 3)
            
            # Select a few important features to display
            if len(real_X.columns) <= 3:
                features_to_display = real_X.columns
            else:
                # Use PCA to identify important features
                pca = PCA(n_components=len(real_X.columns))
                pca.fit(real_scaled)
                importance = np.abs(pca.components_[0])
                indices = np.argsort(importance)[::-1]
                features_to_display = real_X.columns[indices[:3]]
            
            # Plot distributions
            for i, feature in enumerate(features_to_display):
                plt.subplot(2, 2, 3 + i // 3)
                plt.hist(real_X[feature], bins=30, alpha=0.5, density=True, label=f"Real {feature}", color='blue')
                plt.hist(synth_X[feature], bins=30, alpha=0.5, density=True, label=f"Synthetic {feature}", color='red')
                plt.title(f"Distribution: {feature}")
                plt.legend()
        
        plt.tight_layout()
        plt.savefig('comparision.png')
        plt.show()
        plt.close()
        
    def evaluate_with_ml_models(self, test_size=0.2):
        """
        Evaluate how well ML models trained on synthetic data perform on real data
        
        Parameters:
        -----------
        test_size: float
            Proportion of real data to use for testing
        
        Returns:
        --------
        dict
            Performance metrics
        """
        if self.label_column is None:
            print("Cannot evaluate without label column")
            return {}
        
        results = {}
        
        # Split real data into train/test
        X_real = self.real_data.drop(columns=[self.label_column]) 
        y_real = self.real_data[self.label_column]
        
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=test_size, random_state=42, stratify=y_real if len(np.unique(y_real)) < 10 else None
        )
        
        # Prepare synthetic data
        X_synth = self.synthetic_data.drop(columns=[self.label_column])
        y_synth = self.synthetic_data[self.label_column]
        
        # Standardize data
        scaler = StandardScaler()
        X_train_real_scaled = scaler.fit_transform(X_train_real)
        X_test_real_scaled = scaler.transform(X_test_real)
        X_synth_scaled = scaler.transform(X_synth)
        
        # Train on real data, test on real data (baseline)
        xgb_real = xgb.XGBClassifier(random_state=42)
        xgb_real.fit(X_train_real_scaled, y_train_real)
        real_real_preds = xgb_real.predict(X_test_real_scaled)
        results['real_to_real'] = classification_report(y_test_real, real_real_preds, output_dict=True)
        
        # Train on synthetic data, test on real data
        xgb_synth = xgb.XGBClassifier(random_state=42)
        xgb_synth.fit(X_synth_scaled, y_synth)
        synth_real_preds = xgb_synth.predict(X_test_real_scaled)
        results['synth_to_real'] = classification_report(y_test_real, synth_real_preds, output_dict=True)
        
        # Train neural network on real data
        nn_real = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_real_scaled.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(len(np.unique(y_real)), activation='softmax')
        ])
        
        nn_real.compile(optimizer=Adam(learning_rate=0.001), 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'])
        
        nn_real.fit(X_train_real_scaled, y_train_real, 
                    epochs=10, batch_size=32, verbose=0,
                    validation_split=0.1)
        
        nn_real_preds = np.argmax(nn_real.predict(X_test_real_scaled), axis=1)
        results['nn_real_to_real'] = classification_report(y_test_real, nn_real_preds, output_dict=True)
        
        # Train neural network on synthetic data
        nn_synth = Sequential([
            Dense(64, activation='relu', input_shape=(X_synth_scaled.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(len(np.unique(y_synth)), activation='softmax')
        ])
        
        nn_synth.compile(optimizer=Adam(learning_rate=0.001), 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'])
        
        nn_synth.fit(X_synth_scaled, y_synth, 
                    epochs=10, batch_size=32, verbose=0,
                    validation_split=0.1)
        
        nn_synth_preds = np.argmax(nn_synth.predict(X_test_real_scaled), axis=1)
        results['nn_synth_to_real'] = classification_report(y_test_real, nn_synth_preds, output_dict=True)
        
        # Print summary results
        print("\nModel Performance Summary:")
        print("XGBoost trained on real data - Accuracy:", results['real_to_real']['accuracy'])
        print("XGBoost trained on synthetic data - Accuracy:", results['synth_to_real']['accuracy'])
        print("Neural Network trained on real data - Accuracy:", results['nn_real_to_real']['accuracy'])
        print("Neural Network trained on synthetic data - Accuracy:", results['nn_synth_to_real']['accuracy'])
        
        # Visualize confusion matrices
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.heatmap(confusion_matrix(y_test_real, real_real_preds), annot=True, fmt='d', cmap='Blues')
        plt.title("XGBoost: Real → Real")
        
        plt.subplot(2, 2, 2)
        sns.heatmap(confusion_matrix(y_test_real, synth_real_preds), annot=True, fmt='d', cmap='Blues')
        plt.title("XGBoost: Synthetic → Real")
        
        plt.subplot(2, 2, 3)
        sns.heatmap(confusion_matrix(y_test_real, nn_real_preds), annot=True, fmt='d', cmap='Blues')
        plt.title("Neural Network: Real → Real")
        
        plt.subplot(2, 2, 4)
        sns.heatmap(confusion_matrix(y_test_real, nn_synth_preds), annot=True, fmt='d', cmap='Blues')
        plt.title("Neural Network: Synthetic → Real")
        
        plt.tight_layout()
        plt.savefig('summary.png')
        plt.show()
        plt.close()
        
        return results
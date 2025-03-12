import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define file paths
cicids_path = "/Users/harshilpatel/Downloads/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
darknet_path = "/Users/harshilpatel/Downloads/Darknet.CSV"

# Load datasets with stripped column names
df_cicids = pd.read_csv(cicids_path, low_memory=False)
df_darknet = pd.read_csv(darknet_path, low_memory=False, on_bad_lines='skip')

# Strip spaces from column names
df_cicids.columns = df_cicids.columns.str.strip()
df_darknet.columns = df_darknet.columns.str.strip()

# Rename mismatched columns for consistency
column_mapping = {
    'Total Fwd Packet': 'Total Fwd Packets',
    'Total Bwd packets': 'Total Backward Packets',
    'Total Length of Fwd Packet': 'Total Length of Fwd Packets',
    'Total Length of Bwd Packet': 'Total Length of Bwd Packets'
}
df_darknet.rename(columns=column_mapping, inplace=True)

# Define common features
common_features = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 
    'Fwd Packet Length Max', 'Bwd Packet Length Max', 'Label'
]

# Keep only common features
df_cicids = df_cicids[[col for col in common_features if col in df_cicids.columns]]
df_darknet = df_darknet[[col for col in common_features if col in df_darknet.columns]]

# Standardize column names
df_cicids.rename(columns={'Label': 'Traffic_Label'}, inplace=True)
df_darknet.rename(columns={'Label': 'Traffic_Label'}, inplace=True)

# **Fix Encoding Issue**
# Combine both datasets to get all unique labels
all_labels = pd.concat([df_cicids['Traffic_Label'], df_darknet['Traffic_Label']], axis=0)

# Fit encoder on the combined labels
encoder = LabelEncoder()
encoder.fit(all_labels)

# Apply transformation
df_cicids['Traffic_Label'] = encoder.transform(df_cicids['Traffic_Label'])
df_darknet['Traffic_Label'] = encoder.transform(df_darknet['Traffic_Label'])

# **Fix Feature Scaling Issue**
# Recompute num_cols after renaming Label to Traffic_Label
num_cols = [col for col in df_cicids.columns if col != 'Traffic_Label']

# Normalize numerical features
scaler = StandardScaler()
df_cicids[num_cols] = scaler.fit_transform(df_cicids[num_cols])
df_darknet[num_cols] = scaler.transform(df_darknet[num_cols])

# Merge datasets
df_combined = pd.concat([df_cicids, df_darknet], ignore_index=True)

# Save processed dataset
df_combined.to_csv("merged_encrypted_traffic.csv", index=False)
print("Preprocessing complete. Merged dataset saved successfully.")

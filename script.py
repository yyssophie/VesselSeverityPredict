import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Part 1
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 1) Load and clean
train_df = pd.read_csv("/Users/yuanyusi/Downloads/psc_severity_train.csv")

# 2) Remove rows if severity is not Low/Medium/High
valid_labels = ["Low", "Medium", "High"]
train_df = train_df[train_df["annotation_severity"].isin(valid_labels)]
print("After removing invalid severity rows:", train_df.shape)

# 3) Feature Engineering: Add text length as a feature
train_df["annotation_length"] = train_df["def_text"].apply(len)

# 4) Convert severity to weighted scores: Low=1, Medium=2, High=5
severity_map = {"Low": 1, "Medium": 2, "High": 5}
train_df["severity_score"] = train_df["annotation_severity"].map(severity_map)

# 5) Group by (PscInspectionId, deficiency_code), compute average severity score
group_cols = ["PscInspectionId", "deficiency_code"]
grouped_df = train_df.groupby(group_cols)["severity_score"].mean().reset_index(name="avg_score")

# 6) Map average scores to severity with refined thresholds
def map_avg_score_to_severity(avg_score):
    if avg_score < 1.3:
        return "Low"
    elif avg_score < 3:
        return "Medium"
    else:
        return "High"

grouped_df["consensus_severity"] = grouped_df["avg_score"].apply(map_avg_score_to_severity)

print("Grouped consensus DataFrame:")
print(grouped_df.head())
print("Grouped_df shape:", grouped_df.shape)

# 7) Merge consensus back to ALL rows in the original train_df
train_consensus_full = pd.merge(
    train_df,
    grouped_df[["PscInspectionId", "deficiency_code", "avg_score", "consensus_severity"]],
    on=group_cols,
    how="left"
)

print("\nMerged DataFrame (all rows, each with consensus):")
print(train_consensus_full.head())
print("train_consensus_full shape:", train_consensus_full.shape)

# 8) Resample the dataset (SMOTE + Undersampling)
X = train_consensus_full[["severity_score", "avg_score", "annotation_length"]].values
y = train_consensus_full["consensus_severity"]

# Convert categorical severity labels to numerical labels
severity_map_inverse = {"Low": 0, "Medium": 1, "High": 2}
y = y.map(severity_map_inverse)

# Aggressive SMOTE for "High" and balanced undersampling for "Medium"
smote = SMOTE(random_state=42, sampling_strategy={2: 5000})  # Oversample "High" to 1500 samples
rus = RandomUnderSampler(random_state=42, sampling_strategy={1: 2000})  # Undersample "Medium" to 1000 samples

# Apply resampling in sequence
X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

# Convert resampled labels back to original severity names
inverse_severity_map = {v: k for k, v in severity_map_inverse.items()}
y_resampled = pd.Series(y_resampled).map(inverse_severity_map)

# Create a new balanced DataFrame
balanced_df = pd.DataFrame(X_resampled, columns=["severity_score", "avg_score", "annotation_length"])
balanced_df["consensus_severity"] = y_resampled

# 9) Save processed data
output_path = "/Users/yuanyusi/Downloads/processed.csv"
balanced_df.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")





# 7) Visualize the distribution of the final consensus severity
plt.figure(figsize=(6,4))
train_consensus_full['consensus_severity'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Distribution of Consensus Severity (Training)")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.show()





# Part 2: Training Preparation 
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Load a lightweight and fast sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient and optimized for CPU

# Generate embeddings for the `def_text` field
print("Generating embeddings for training data...")
train_consensus_full['embedding'] = model.encode(
    train_consensus_full['def_text'].tolist(),
    show_progress_bar=True
).tolist()

# Encode severity labels
label_encoder = LabelEncoder()
train_consensus_full['severity_encoded'] = label_encoder.fit_transform(train_consensus_full['consensus_severity'])

# Extract features (X) and labels (y)
X = np.array(train_consensus_full['embedding'].tolist())  # Convert embeddings to NumPy array
y = train_consensus_full['severity_encoded']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Logistic Regression classifier (fast on CPU)
print("Training the classifier...")
clf = LogisticRegression(random_state=42, max_iter=500)
clf.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = clf.predict(X_val)

# Print classification report
print("Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

# Print confusion matrix (optional)
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# Part 2: Testing
test_df = pd.read_csv("/Users/yuanyusi/Downloads/psc_severity_test.csv")

# Generate embeddings for the test data
print("Generating embeddings for test data...")
test_df['embedding'] = model.encode(
    test_df['def_text'].tolist(),
    show_progress_bar=True
).tolist()

# Predict severity for the test data
X_test = np.array(test_df['embedding'].tolist())
test_df['severity_encoded'] = clf.predict(X_test)

# Decode severity to original labels
test_df['predicted_severity'] = label_encoder.inverse_transform(test_df['severity_encoded'])

# Save the predictions to a CSV file
output_file = "/Users/yuanyusi/Downloads/test_predictions.csv"
test_df[['PscInspectionId', 'deficiency_code', 'predicted_severity']].to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")


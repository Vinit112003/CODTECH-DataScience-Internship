import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv("sample_data.csv")

print("Original Data:")
print(data.head())

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Create preprocessing pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Apply transformations
X_processed = pipeline.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print("Pipeline executed successfully!")

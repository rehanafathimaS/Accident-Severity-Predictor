import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
# Unga dataset name check pannikonga (safety analysis.CSV)
df = pd.read_csv('safety analysis.CSV')

# 2. Data Cleaning
# Remove non-numeric columns that are not useful for prediction
df_clean = df.drop(columns=['Accident_ID', 'Date', 'Time'])

# 3. Encoding categorical data
encoders = {}
for col in df_clean.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    encoders[col] = le

# 4. Features and Target
X = df_clean.drop('Severity', axis=1)
y = df_clean['Severity']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 6. Save as Pickle
with open('safety_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'encoders': encoders,
        'feature_names': feature_names
    }, f)

print("Model trained and saved as safety_model.pkl!")
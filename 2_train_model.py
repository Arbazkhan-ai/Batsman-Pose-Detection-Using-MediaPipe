import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('cricket_shots_data.csv', header=None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
model = pipeline.fit(X_train, y_train)
print('Model Accuracy:', model.score(X_test, y_test))

with open('cricket_shot_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Model saved successfully!")
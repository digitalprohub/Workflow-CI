import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train():
    mlflow.set_experiment("BNPL_CI_Automation")
    
    # Enable Autologging
    mlflow.autolog()
    
    # Load Preprocessed Data
    train_df = pd.read_csv('BNPL_preprocessing/train_clean.csv')
    test_df = pd.read_csv('BNPL_preprocessing/test_clean.csv')
    
    X_train = train_df.drop(columns=['Default_Risk'])
    y_train = train_df['Default_Risk']
    X_test = test_df.drop(columns=['Default_Risk'])
    y_test = test_df['Default_Risk']
    
    with mlflow.start_run(run_name="CI_ReTraining_Run"):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"CI Training Completed. Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

if __name__ == "__main__":
    train()
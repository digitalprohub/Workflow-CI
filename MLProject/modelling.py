import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def train():
    # 1. Autolog Wajib
    mlflow.autolog()
    
    # 2. Path Relatif ke Preprocessing Data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'BNPL_preprocessing', 'train_clean.csv')
    test_path = os.path.join(base_dir, 'BNPL_preprocessing', 'test_clean.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['Default_Risk'])
    y_train = train_df['Default_Risk']
    X_test = test_df.drop(columns=['Default_Risk'])
    y_test = test_df['Default_Risk']
    
    # 3. Model Training
    # Jika dijalankan lewat `mlflow run`, MLflow secara otomatis mengelola pencatatan run.
    # Kita tidak perlu memanggil mlflow.start_run() jika MLFLOW_RUN_ID ada di environment.
    model = RandomForestClassifier(random_state=42)
    
    if "MLFLOW_RUN_ID" in os.environ:
        # Dieksekusi via `mlflow run` (CI Environment)
        model.fit(X_train, y_train)
    else:
        # Dieksekusi langsung via `python modelling.py`
        mlflow.set_experiment("BNPL_CI_Automation")
        with mlflow.start_run(run_name="CI_ReTraining_Run"):
            model.fit(X_train, y_train)

if __name__ == '__main__':
    train()
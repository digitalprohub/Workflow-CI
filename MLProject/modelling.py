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
    # Jika dipanggil via `mlflow run`, run sudah aktif, jika tidak maka buat run baru
    active_run = mlflow.active_run()
    if active_run:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
    else:
        mlflow.set_experiment("BNPL_CI_Automation")
        with mlflow.start_run(run_name="CI_ReTraining_Run"):
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

if __name__ == '__main__':
    train()
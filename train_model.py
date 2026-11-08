import pandas as pd
import numpy as np
import os
import glob
import kagglehub
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 7150

def load_data():
    print("Загрузка датасета...")
    try:
        path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if csv_files:
            file_path = csv_files[0]
            print(f"Файл найден: {file_path}")
            return pd.read_csv(file_path)
    except Exception as e:
        print(f"Ошибка загрузки через kagglehub: {e}")
    
    local_path = "diabetes_prediction_dataset.csv"
    if os.path.exists(local_path):
        print(f"Загрузка локального файла: {local_path}")
        return pd.read_csv(local_path)
    
    raise FileNotFoundError("Не удалось найти датасет. Скачайте его вручную с Kaggle.")

def preprocess_data(df):
    print("Предобработка данных...")
    
    df = df.drop_duplicates().reset_index(drop=True)
    
    cat_cols = ['gender', 'smoking_history']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    X = df_encoded.drop(columns=['diabetes'])
    y = df_encoded['diabetes']
    
    return X, y

def train_and_save_model():
    df = load_data()
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Обучение логистической регрессии...")
    model = LogisticRegression(
        random_state=SEED,
        class_weight='balanced',
        max_iter=1000,
        C=1.0
    )
    model.fit(X_train_scaled, y_train)
    
    print("Сохранение модели и трансформеров...")
    joblib.dump(model, 'diabetes_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    joblib.dump(list(X.columns), 'model_features.joblib')
    
    print("Готово! Файлы сохранены: diabetes_model.joblib, scaler.joblib, model_features.joblib")

if __name__ == "__main__":
    train_and_save_model()
import joblib
import pickle
import pandas as pd
from datetime import datetime

def load_model():
    model = joblib.load('xgboost_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    state_encoder = joblib.load('state_encoder.pkl')
    with open('feature_info.pkl', 'rb') as f:
        feature_info = pickle.load(f)
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return model, scaler, state_encoder, feature_info, metadata

def predict_for_date(date, model, state_encoder, scaler, feature_info):
    date = pd.to_datetime(date)
    predictions = {}
    for state in state_encoder.classes_:
        state_encoded = state_encoder.transform([state])[0]
        features = pd.DataFrame({
            'State_Encoded': [state_encoded],
            'DayOfYear': [date.dayofyear],
            'Month': [date.month],
            'DayOfWeek': [date.dayofweek],
            'Season': [0 if date.month in [12, 1, 2] else 1 if date.month in [3, 4, 5] else 2 if date.month in [6, 7, 8] else 3]
        })
        features_scaled = scaler.transform(features[feature_info['feature_columns']])
        pred = model.predict(features_scaled)[0]
        predictions[state] = pred
    return predictions

def main():
    print("Загрузка модели XGBoost...")
    model, scaler, state_encoder, feature_info, metadata = load_model()
    print(f"Модель поддерживает {len(metadata['states'])} штатов")
    print(f"Test RMSE: {metadata['test_rmse']:.2f} MW")
    print(f"Test MAPE: {metadata['test_mape']:.2f}%")
    
    target_date = input("Введите дату (гггг-мм-дд): ")
    predictions = predict_for_date(target_date, model, state_encoder, scaler, feature_info)
    print(f"
Прогноз потребления на {target_date}:")
    for state, consumption in predictions.items():
        print(f"{state}: {consumption:.2f} MW")

if __name__ == "__main__":
    main()

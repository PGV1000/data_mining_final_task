# Power Consumption Prediction App

Streamlit-приложение для прогнозирования потребления электроэнергии в штатах Индии на основе XGBoost модели.

## Требования

- Python 3.9+
- streamlit
- joblib
- pandas
- numpy
- geopandas
- folium
- streamlit-folium
- plotly
- scikit-learn
- xgboost

## Установка

```bash
pip install streamlit joblib pandas numpy geopandas folium streamlit-folium plotly scikit-learn xgboost
```

## Структура файлов

```
project/
├── app.py                          # Основное приложение
├── india_states_updated.json       # GeoJSON файл с границами штатов
└── xgboost_power_consumption_model/
    ├── xgboost_model.pkl
    ├── feature_scaler.pkl
    ├── state_encoder.pkl
    ├── feature_info.pkl
    ├── model_metadata.pkl
    └── processed_data.csv
```

## Запуск

```bash
streamlit run app.py
```

## Использование

1. Выберите дату в боковой панели
2. Выберите штат или "Все штаты"
3. Нажмите "Predict"
4. Просмотрите результаты на карте и в таблице

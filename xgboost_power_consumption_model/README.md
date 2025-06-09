# XGBoost Power Consumption Forecasting Model

## Обзор модели
- **Тип модели**: XGBoost
- **Штаты**: 33 индийских штатов
- **Период данных**: 2019-01-02 00:00:00 - 2020-12-05 00:00:00
- **Общее количество записей**: 16,434
- **Признаки**: State_Encoded, DayOfYear, Month, DayOfWeek, Season

## Метрики производительности
### RMSE
- **Тренировочная выборка**: 22.03 MW
- **Тестовая выборка**: 22.20 MW

### MAE
- **Тренировочная выборка**: 13.53 MW
- **Тестовая выборка**: 13.56 MW

### MAPE
- **Тренировочная выборка**: 58.38%
- **Тестовая выборка**: 61.03%

### Анализ обобщения
- **Test/Train RMSE Ratio**: 1.01
- **Статус переобучения**: ✓ Хорошее обобщение

## Признаки
- **State_Encoded**: Закодированное название штата
- **DayOfYear**: День года
- **Month**: Месяц
- **DayOfWeek**: День недели
- **Season**: Сезон (0: Зима, 1: Весна, 2: Лето, 3: Осень)

## Структура файлов
```
xgboost_power_consumption_model/
├── xgboost_model.pkl        # Обученная модель XGBoost
├── feature_scaler.pkl       # Масштабировщик признаков
├── state_encoder.pkl        # Энкодер штатов
├── feature_info.pkl         # Информация о признаках
├── model_metadata.pkl       # Метаданные модели
├── processed_data.csv       # Обработанные данные
├── predict.py               # Скрипт предсказания
└── README.md                # Документация
```

## Загрузка и использование модели
```python
import joblib
import pickle

model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
state_encoder = joblib.load('state_encoder.pkl')
with open('feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)
with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
```

## Пример предсказания
```python
from predict import predict_for_date
target_date = '2025-06-10'
predictions = predict_for_date(target_date, model, state_encoder, scaler, feature_info)
for state, consumption in predictions.items():
    print(f"West Bengal: 129.05 MW")
```

## Статистика данных
- **Общее количество записей**: 16,434
- **Тренировочная выборка**: 13,147
- **Тестовая выборка**: 3,287
- **Количество признаков**: 5

## Ограничения модели
- Не учитывает исторические тренды (лаговые признаки).
- Точность зависит от сезонных и временных паттернов.
- Данные 2013–2022 годов, для новых данных может потребоваться переобучение.

Дата создания: 2025-06-09 09:16:20

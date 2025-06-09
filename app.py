import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from datetime import datetime
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class PowerConsumptionPredictor:
    def __init__(self, model_dir="xgboost_power_consumption_model"):
        """Инициализация предиктора с загрузкой компонентов XGBoost модели"""
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.state_encoder = None
        self.metadata = None
        self.feature_info = None
        self.processed_data = None
    
    @st.cache_resource
    def load_model_components(_self):
        """Загрузка компонентов модели с кешированием"""
        try:
            # Загрузка модели
            model_path = os.path.join(_self.model_dir, "xgboost_model.pkl")
            _self.model = joblib.load(model_path)
            
            # Загрузка масштабировщика
            scaler_path = os.path.join(_self.model_dir, "feature_scaler.pkl")
            _self.scaler = joblib.load(scaler_path)
            
            # Загрузка энкодера штатов
            state_encoder_path = os.path.join(_self.model_dir, "state_encoder.pkl")
            _self.state_encoder = joblib.load(state_encoder_path)
            
            # Загрузка метаданных
            metadata_path = os.path.join(_self.model_dir, "model_metadata.pkl")
            with open(metadata_path, 'rb') as f:
                _self.metadata = pickle.load(f)
            
            # Загрузка информации о признаках
            feature_info_path = os.path.join(_self.model_dir, "feature_info.pkl")
            with open(feature_info_path, 'rb') as f:
                _self.feature_info = pickle.load(f)
            
            # Загрузка обработанных данных
            processed_data_path = os.path.join(_self.model_dir, "processed_data.csv")
            _self.processed_data = pd.read_csv(processed_data_path)
            _self.processed_data['Date'] = pd.to_datetime(_self.processed_data['Date'])
            
            return True
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {str(e)}")
            return False
    
    def predict_for_date(self, date):
        """Предсказание потребления для всех штатов на заданную дату"""
        date = pd.to_datetime(date)
        predictions = {}
        
        for state in self.state_encoder.classes_:
            try:
                state_encoded = self.state_encoder.transform([state])[0]
                features = pd.DataFrame({
                    'State_Encoded': [state_encoded],
                    'DayOfYear': [date.dayofyear],
                    'Month': [date.month],
                    'DayOfWeek': [date.dayofweek],
                    'Season': [0 if date.month in [12, 1, 2] else 1 if date.month in [3, 4, 5] else 2 if date.month in [6, 7, 8] else 3]
                })
                features_scaled = self.scaler.transform(features[self.feature_info['feature_columns']])
                pred = self.model.predict(features_scaled)[0]
                predictions[state] = pred
            except Exception as e:
                st.warning(f"Ошибка предсказания для {state}: {e}")
                predictions[state] = None
        
        return predictions
    
    def create_prediction_chart(self, state_name, date):
        """Создание графика с историческими данными и предсказанием"""
        if self.processed_data is None:
            return None
        
        try:
            predictions = self.predict_for_date(date)
            pred_value = predictions.get(state_name)
            if pred_value is None:
                raise ValueError("Не удалось получить предсказание")
        except Exception as e:
            st.error(f"Ошибка создания графика: {e}")
            return None
        
        # Исторические данные для штата
        state_data = self.processed_data[self.processed_data['States'] == state_name].copy()
        state_data = state_data.sort_values('Date')
        
        # Последние 60 дней
        recent_data = state_data.tail(60)
        
        # Даты для графика
        last_date = recent_data['Date'].max()
        pred_date = pd.to_datetime(date)
        
        # Создание графика
        fig = go.Figure()
        
        # Исторические данные
        fig.add_trace(go.Scatter(
            x=recent_data['Date'],
            y=recent_data['Usage'],
            mode='lines',
            name='Исторические данные',
            line=dict(color='blue', width=2)
        ))
        
        # Предсказание
        fig.add_trace(go.Scatter(
            x=[pred_date],
            y=[pred_value],
            mode='markers',
            name='Предсказание',
            marker=dict(size=10, color='red')
        ))
        
        # Соединительная линия
        if len(recent_data) > 0 and last_date <= pred_date:
            fig.add_trace(go.Scatter(
                x=[last_date, pred_date],
                y=[recent_data['Usage'].iloc[-1], pred_value],
                mode='lines',
                name='Соединение',
                line=dict(color='orange', width=2, dash='dash'),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f'Прогноз потребления электроэнергии - {state_name}',
            xaxis_title='Дата',
            yaxis_title='Потребление (MW)',
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def get_model_info(self):
        """Получение информации о модели"""
        if not self.metadata:
            return {}
        
        return {
            'Тип модели': self.metadata.get('model_type', 'XGBoost'),
            'Количество штатов': len(self.metadata.get('states', [])),
            'Период данных': f"{self.metadata.get('date_range', {}).get('start', 'N/A')} - {self.metadata.get('date_range', {}).get('end', 'N/A')}",
            'RMSE тест': f"{self.metadata.get('test_rmse', 0):.2f} MW",
            'MAE тест': f"{self.metadata.get('test_mae', 0):.2f} MW",
            'MAPE тест': f"{self.metadata.get('test_mape', 0):.2f}%",
            'Коэффициент обобщения': f"{self.metadata.get('test_train_ratio', 0):.2f}",
            'Создана': self.metadata.get('created_at', 'N/A')
        }

# Функция загрузки GeoJSON
@st.cache_data
def load_geojson():
    """Загрузка GeoJSON файла с границами штатов"""
    try:
        possible_paths = [
            "india_states_updated.json",
            "data/india_states_updated.json",
            "../data/india_states_updated.json"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return gpd.read_file(path)
        st.warning("GeoJSON файл не найден. Карта будет недоступна.")
        return None
    except Exception as e:
        st.error(f"Ошибка загрузки GeoJSON: {e}")
        return None

def get_continuous_color(value, min_val, max_val):
    """Получение непрерывного цвета на основе значения"""
    if value is None or min_val is None or max_val is None:
        return '#808080'  # Серый для отсутствующих данных
    
    # Нормализация значения от 0 до 1
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    
    # Создание цветовой карты от синего (низкие значения) до красного (высокие значения)
    # Используем более контрастную цветовую схему
    if normalized <= 0.25:
        # Синий к голубому
        r = int(0 + normalized * 4 * 100)
        g = int(100 + normalized * 4 * 155)
        b = 255
    elif normalized <= 0.5:
        # Голубой к зеленому
        factor = (normalized - 0.25) * 4
        r = int(100 - factor * 100)
        g = 255
        b = int(255 - factor * 255)
    elif normalized <= 0.75:
        # Зеленый к желтому
        factor = (normalized - 0.5) * 4
        r = int(factor * 255)
        g = 255
        b = 0
    else:
        # Желтый к красному
        factor = (normalized - 0.75) * 4
        r = 255
        g = int(255 - factor * 255)
        b = 0
    
    return f'#{r:02x}{g:02x}{b:02x}'

def create_folium_map(predictions, geo_df):
    """Создание интерактивной карты с Folium"""
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="cartodbpositron")
    
    if geo_df is not None:
        # Получение минимального и максимального значений для нормализации цветов
        valid_predictions = [v for v in predictions.values() if v is not None]
        if valid_predictions:
            min_val = min(valid_predictions)
            max_val = max(valid_predictions)
        else:
            min_val = max_val = None
        
        for _, row in geo_df.iterrows():
            state_name = row['NAME_1']
            usage = predictions.get(state_name)
            color = get_continuous_color(usage, min_val, max_val)
            tooltip_text = f"{state_name}: {usage:.2f} MW" if usage is not None else f"{state_name}: Нет данных"
            
            # Добавление полигона
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x, fillColor=color: {
                    'fillColor': fillColor,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7
                },
                tooltip=tooltip_text
            ).add_to(m)
            
            # Добавление текста в центр полигона
            if usage is not None:
                # Вычисление центроида полигона
                if hasattr(row['geometry'], 'centroid'):
                    centroid = row['geometry'].centroid
                    lat, lon = centroid.y, centroid.x
                    
                    # Добавление маркера с текстом
                    folium.Marker(
                        location=[lat, lon],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 10px; color: black; font-weight: bold; text-align: center; background-color: rgba(255,255,255,0.8); border-radius: 3px; padding: 2px;">{usage:.1f}</div>',
                            icon_size=(40, 20),
                            icon_anchor=(20, 10)
                        )
                    ).add_to(m)
    
    # Убираем легенду (закомментировано)
    # legend_html = '''...'''
    # m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def main():
    st.set_page_config(page_title="Прогноз потребления электроэнергии", page_icon="⚡", layout="wide")
    st.title("⚡ Прогноз потребления электроэнергии в Индии")
    st.markdown("*Модель XGBoost для предсказания на основе даты*")
    st.markdown("---")
    
    # Инициализация предиктора
    if 'predictor' not in st.session_state:
        st.session_state.predictor = PowerConsumptionPredictor()
    
    predictor = st.session_state.predictor
    
    # Загрузка модели
    if predictor.model is None:
        with st.spinner("Загрузка модели XGBoost..."):
            success = predictor.load_model_components()
        if not success:
            st.error("Не удалось загрузить модель. Проверьте, что папка 'xgboost_power_consumption_model' содержит файлы: xgboost_model.pkl, feature_scaler.pkl, state_encoder.pkl, feature_info.pkl, model_metadata.pkl, processed_data.csv")
            st.stop()
        st.success("✅ Модель XGBoost загружена!")
    
    # Загрузка GeoJSON
    geo_df = load_geojson()
    
    # Боковая панель
    st.sidebar.header("🔧 Настройки прогноза")
    
    # Информация о модели
    model_info = predictor.get_model_info()
    with st.sidebar.expander("ℹ️ Информация о модели"):
        for key, value in model_info.items():
            st.text(f"{key}: {value}")
    
    # Ввод даты
    target_date = st.sidebar.date_input(
        "Выберите дату:",
        value=datetime.now().date(),
        min_value=datetime(2013, 1, 1),
        max_value=datetime(2025, 12, 31),
        help="Выберите дату для прогноза"
    )
    
    # Выбор штата
    available_states = list(predictor.state_encoder.classes_) if predictor.state_encoder else []
    selected_state = st.sidebar.selectbox(
        "Выберите штат:",
        options=['Все штаты'] + sorted(available_states),
        help="Выберите штат или 'Все штаты' для общего прогноза"
    )
    
    # Кнопка для прогноза
    if st.sidebar.button("🚀 Predict", type="primary", use_container_width=True):
        st.session_state.update_predictions = True
    
    # Генерация прогнозов
    if st.session_state.get('update_predictions', False):
        with st.spinner("Генерация прогнозов..."):
            predictions = predictor.predict_for_date(target_date)
            st.session_state.predictions = predictions
        st.session_state.update_predictions = False
    
    # Отображение результатов
    if 'predictions' in st.session_state and st.session_state.predictions:
        if selected_state == 'Все штаты':
            st.subheader("🗺️ Карта прогноза потребления")
            if geo_df is not None:
                folium_map = create_folium_map(st.session_state.predictions, geo_df)
                st_folium(folium_map, width=700, height=500, key="folium_map")
            else:
                st.warning("Карта недоступна - GeoJSON файл не найден")
            
            st.subheader("📊 Прогнозы по штатам")
            pred_data = [
                {
                    'Штат': state,
                    'Прогноз (MW)': f"{pred:.2f}" if pred is not None else 'N/A',
                    'Категория': 'Низкое' if pred is not None and pred < 100 else 'Среднее' if pred is not None and pred < 300 else 'Высокое' if pred is not None and pred < 500 else 'Очень высокое' if pred is not None else 'Нет данных'
                }
                for state, pred in st.session_state.predictions.items()
            ]
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True)
            
            valid_predictions = [v for v in st.session_state.predictions.values() if v is not None]
        else:
            st.subheader(f"📈 Прогноз для штата {selected_state}")
            fig = predictor.create_prediction_chart(selected_state, target_date)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Прогноз")
            forecast_df = pd.DataFrame({
                'Дата': [target_date.strftime('%Y-%m-%d')],
                'День недели': [target_date.strftime('%A')],
                'Прогноз (MW)': [round(st.session_state.predictions[selected_state], 2)] if st.session_state.predictions[selected_state] is not None else ['N/A']
            })
            st.dataframe(forecast_df, use_container_width=True)
    
    # Инструкции
    with st.expander("📖 Инструкции по использованию"):
        st.markdown("""
        ### О модели:
        - **Тип**: XGBoost
        - **Признаки**: Штат, день года, месяц, день недели, сезон
        - **Данные**: 2013–2022 годы, индийские штаты
        """)

if __name__ == "__main__":
    main()
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import joblib


st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Оценка риска сахарного диабета")
st.markdown("""
Это учебное приложение для скрининговой оценки риска диабета на основе клинических показателей.
Введите данные пациента и нажмите **«Предсказать»**.""")
st.divider()

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('diabetes_model.joblib')
        scaler = joblib.load('scaler.joblib')
        features = joblib.load('model_features.joblib')
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None

model, scaler, expected_features = load_artifacts()

if model is None:
    st.error("Ошибка: Файлы модели не найдены. Сначала запустите `python train_model.py` в корневой папке.")
    st.stop()

st.sidebar.header("Ввод данных пациента")

if st.sidebar.button("Заполнить тестовым примером"):
    st.session_state.age = 45
    st.session_state.gender = "Female"
    st.session_state.bmi = 28.5
    st.session_state.hypertension = 0
    st.session_state.heart_disease = 0
    st.session_state.smoking_history = "never"
    st.session_state.hba1c = 6.8
    st.session_state.glucose = 140

with st.sidebar.form("patient_form"):
    age = st.number_input("Возраст", min_value=1, max_value=120, value=st.session_state.get('age', 45))
    gender = st.selectbox("Пол", options=["Female", "Male", "Other"], index=0)
    bmi = st.number_input("ИМТ (Индекс массы тела)", min_value=10.0, max_value=70.0, value=st.session_state.get('bmi', 25.0), step=0.1)
    
    col1, col2 = st.columns(2)
    with col1:
        hypertension = st.checkbox("Гипертония", value=False)
    with col2:
        heart_disease = st.checkbox("Болезни сердца", value=False)
        
    smoking = st.selectbox("Статус курения", options=["never", "former", "current", "not current", "ever", "No Info"])
    hba1c = st.number_input("Уровень HbA1c (%)", min_value=3.0, max_value=15.0, value=st.session_state.get('hba1c', 5.5), step=0.1)
    glucose = st.number_input("Уровень глюкозы (мг/дл)", min_value=50, max_value=400, value=st.session_state.get('glucose', 100))
    
    submit_button = st.form_submit_button(label="Предсказать риск", use_container_width=True)
    
    if st.form_submit_button("Очистить форму", use_container_width=True):
        st.session_state.clear()
        st.rerun()

if submit_button:
    input_data = {
        "age": [float(age)],
        "gender": [gender],
        "bmi": [float(bmi)],
        "hypertension": [int(hypertension)],
        "heart_disease": [int(heart_disease)],
        "smoking_history": [smoking],
        "HbA1c_level": [float(hba1c)],
        "blood_glucose_level": [int(glucose)]
    }
    
    try:
        df_input = pd.DataFrame(input_data)
        df_encoded = pd.get_dummies(df_input, columns=['gender', 'smoking_history'])
        
        df_final = df_encoded.reindex(columns=expected_features, fill_value=0)
        

        X_scaled = scaler.transform(df_final)
        prob = model.predict_proba(X_scaled)[0][1]
        prediction = int(prob >= 0.5)
        risk_class = "Высокий риск" if prediction == 1 else "Низкий риск"
        

        st.session_state.history.append({
            "Возраст": age, "ИМТ": bmi, "HbA1c": hba1c, 
            "Глюкоза": glucose, "Вероятность": f"{prob*100:.1f}%", 
            "Вердикт": risk_class
        })
            
        st.subheader("Результат оценки")
            
        if prediction == 1:
            st.error(f"### {risk_class}\nВероятность диабета: **{prob*100:.1f}%**")
        else:
            st.success(f"### {risk_class}\nВероятность диабета: **{prob*100:.1f}%**")
                
        st.markdown("#### 1. Шкала вероятности")
        st.progress(prob)
        st.caption(f"Модель предсказала вероятность {prob*100:.1f}%. Порог отсечения для высокого риска составляет 50%.")
            
        st.markdown("#### 2. Сравнение с нормами")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            
        ax1.barh(['Пациент'], [hba1c], color='#1f77b4', height=0.5)
        ax1.axvline(x=5.7, color='orange', linestyle='--', label='Преддиабет (5.7%)')
        ax1.axvline(x=6.5, color='red', linestyle='--', label='Диабет (6.5%)')
        ax1.set_xlim(3, 10)
        ax1.set_title('Уровень HbA1c (%)')
        ax1.legend(loc='lower right', fontsize='small')
            
        ax2.barh(['Пациент'], [glucose], color='#ff7f0e', height=0.5)
        ax2.axvline(x=100, color='orange', linestyle='--', label='Преддиабет (100)')
        ax2.axvline(x=126, color='red', linestyle='--', label='Диабет (126)')
        ax2.set_xlim(50, 250)
        ax2.set_title('Уровень глюкозы (мг/дл)')
        ax2.legend(loc='lower right', fontsize='small')
            
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Произошла ошибка при обработке данных: {e}")

st.divider()
st.subheader("История предсказаний (за текущую сессию)")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    if st.button("Очистить историю"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("История пуста. Сделайте первое предсказание.")
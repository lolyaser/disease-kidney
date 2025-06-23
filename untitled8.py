# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

# تحميل البيانات وتدريب النماذج
@st.cache_data
def load_and_train_models():
    df = pd.read_csv('kidney_disease (2).csv')
    df['classification'] = df['classification'].apply(lambda x: 1 if x == 'ckd' else 0)

    features = ['age', 'bp', 'al', 'su', 'sc', 'sod', 'pot', 'hemo', 'bgr', 'bu']

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X = imputer.fit_transform(df[features])
    X = scaler.fit_transform(X)
    y = df['classification']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_model = SVC(probability=True)
    svm_model.fit(X_train, y_train)

    ckd_data = df[df['classification'] == 1]
    X_ckd = imputer.transform(ckd_data[features])
    X_ckd = scaler.transform(X_ckd)
    y_sessions = np.random.randint(1, 4, size=len(ckd_data))

    sessions_model = RandomForestRegressor(n_estimators=100)
    sessions_model.fit(X_ckd, y_sessions)

    return df, features, imputer, scaler, svm_model, sessions_model

# حساب eGFR
def calculate_gfr(age, sc, gender='female'):
    try:
        age = float(age)
        sc = float(sc)
        if gender.lower() == 'male':
            k = 0.9
            alpha = -0.302
            multiplier = 1
        else:
            k = 0.7
            alpha = -0.241
            multiplier = 1.012
        gfr = 142 * (min(sc/k, 1)**alpha) * (max(sc/k, 1)**-1.2) * (0.9938**age) * multiplier
        return max(gfr, 1)
    except:
        return None

# تصنيف المرحلة
def classify_stage(gfr):
    if gfr is None:
        return "Unable to calculate"
    elif gfr >= 90:
        return "Stage 1 (Normal or high GFR)"
    elif 60 <= gfr < 90:
        return "Stage 2 (Mildly decreased GFR)"
    elif 45 <= gfr < 60:
        return "Stage 3a (Mild to moderate decrease)"
    elif 30 <= gfr < 45:
        return "Stage 3b (Moderate to severe decrease)"
    elif 15 <= gfr < 30:
        return "Stage 4 (Severely decreased GFR)"
    else:
        return "Stage 5 (Kidney failure)"

# ----------------------- واجهة Streamlit -----------------------
st.title("Kidney Disease Diagnosis System")
df, features, imputer, scaler, svm_model, sessions_model = load_and_train_models()

st.header("🧾 Enter Patient Data")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature.upper()}:", step=0.1, format="%.2f")

gender = st.radio("Gender", ['female', 'male'])

# زر التشخيص
if st.button("🔍 Diagnose"):
    try:
        input_data = [user_input[f] for f in features]
        data = imputer.transform([input_data])
        data = scaler.transform(data)

        pred = svm_model.predict(data)[0]
        proba = max(svm_model.predict_proba(data)[0]) * 100

        gfr = calculate_gfr(user_input['age'], user_input['sc'], gender)
        stage = classify_stage(gfr)

        status = "Chronic (CKD)" if pred == 1 else "Not Chronic"
        st.subheader("✅ Diagnosis Result")
        st.write(f"**Status:** {status}")
        st.write(f"**Confidence:** {proba:.2f}%")
        st.write(f"**eGFR:** {gfr:.2f} mL/min/1.73m²")
        st.write(f"**Stage:** {stage}")

        if gfr is not None:
            if gfr < 15:
                dialysis = "Dialysis Required (Stage 5 - Kidney Failure)"
                reason = "eGFR is below 15, indicating complete kidney failure."
                sessions = "Recommended: 3 times/week"
            elif 15 <= gfr < 30:
                dialysis = "Dialysis Preparation (Stage 4)"
                reason = "eGFR is between 15 and 30, kidneys are severely impaired."
                sessions = "Recommended: 1–2 times/week"
            elif 30 <= gfr < 60:
                dialysis = "Usually Not Required (Stage 3)"
                reason = "eGFR between 30 and 60; dialysis not typically needed."
                sessions = "If already undergoing dialysis, continue as advised"
            else:
                dialysis = "Not Required"
                reason = "Kidney function is within normal or mild range."
                sessions = "N/A"

            similar_patients = df[
                (df['classification'] == 1) &
                (df['sc'] < user_input['sc'] + 0.3) & (df['sc'] > user_input['sc'] - 0.3) &
                (df['hemo'] < user_input['hemo'] + 1) & (df['hemo'] > user_input['hemo'] - 1)
            ]

            if len(similar_patients) > 0 and gfr > 30:
                dialysis += " (Note: Similar patients received dialysis)"
                sessions = "Please consult your doctor"

            st.write(f"**Dialysis:** {dialysis}")
            st.write(f"**Sessions:** {sessions}")
            st.info(f"**Reason:** {reason}")
        else:
            st.warning("⚠️ eGFR could not be calculated. Please check the inputs.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# زر تحميل بيانات عشوائية
if st.button("🎲 Load Random Patient"):
    patient = df.sample(1).iloc[0]
    st.write("### Random Patient Data")
    st.write(patient[features])

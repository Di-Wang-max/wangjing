import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components
from sklearn.calibration import CalibratedClassifierCV
st.markdown('<h2 style="font-size:20px;">XGBoost Model for Postoperative Thrombosis</h2>', unsafe_allow_html=True)

Age = st.number_input("Age (Year):")
D3Dimer = st.number_input("D3Dimer (μg/mL):")
D5Dimer = st.number_input("D5Dimer (μg/mL):")
FDP = st.number_input("FDP (μg/mL):")
Lymphocyte = st.number_input("Lymphocyte (10^9/L):")
PLT = st.number_input("PLT (10^9/L):")
PrevWF = st.number_input("PrevWF (ng/mL):")
vWFD3 = st.number_input("vWFD3 (ng/mL):")
Operationtime = st.number_input("Operation time (min):")
Anticoagulation = st.selectbox('Anticoagulation', ['No', 'Yes'])
Differentiation = st.selectbox('Differentiation', ['low',"medium",'high'])
Differentiationmap = {'low': 0, 'medium': 1, 'high': 2}
Differentiation = Differentiationmap[Differentiation]
Anticoagulation = 1 if Anticoagulation == 'Yes' else 0

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    XGB = joblib.load("XGB.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Store inputs into dataframe
    input_numerical = np.array([Age,Anticoagulation,D3Dimer,D5Dimer,Differentiation,FDP,Lymphocyte,Operationtime,PLT,PrevWF,vWFD3]).reshape(1, -1)
    feature_names  = ["Age", "Anticoagulation", "D3Dimer","D5Dimer","Differentiation","FDP","Lymphocyte","Operationtime","PLT","PrevWF","vWFD3"]
    input_numericalyuan = pd.DataFrame(input_numerical, columns=feature_names)
    input_numerical = pd.DataFrame(input_numerical, columns=feature_names)

    input_numerical[['D5Dimer','vWFD3','D3Dimer','PrevWF','Age','PLT','FDP','Lymphocyte','Operationtime']] = scaler.transform(input_numerical[['D5Dimer','vWFD3','D3Dimer','PrevWF','Age','PLT','FDP','Lymphocyte','Operationtime']])


    calibrated_clf = CalibratedClassifierCV(XGB, method="isotonic", cv=10)
    calibrated_clf.fit(X_train, y_train)
    prediction_proba = calibrated_clf.predict_proba(input_numerical)[:, 1]
    prediction_proba = (prediction_proba * 100).round(2)
    st.markdown("## **Prediction Probabilities (%)**")
    for prob in target_class_proba_percent:
        st.markdown(f"**{prob:.3f}%**")

  
    explainer = shap.TreeExplainer(XGB)
    shap_values = explainer.shap_values(input_numerical)
    
    st.write("### SHAP Value Force Plot")
    shap.initjs()
    force_plot_visualizer = shap.plots.force(
        explainer.expected_value, shap_values, input_numericalyuan)
    shap.save_html("force_plot.html", force_plot_visualizer)

    with open("force_plot.html", "r", encoding="utf-8") as html_file:
        html_content = html_file.read()

    components.html(html_content, height=400)

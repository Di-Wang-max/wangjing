import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components
from sklearn.calibration import CalibratedClassifierCV
st.markdown('<h2 style="font-size:20px;">XGBoost Model for Postoperative Thrombosis</h2>', unsafe_allow_html=True)

Age = st.number_input("Age (Year):")
D_dimer_D3 = st.number_input("D-dimer(D3) (μg/mL):")
D_dimer_D5 = st.number_input("D-dimer(D5) (μg/mL):")
FDP = st.number_input("FDP (mg/mL):")
Lymphocyte = st.number_input("Lymphocyte count (10^9/L):")
PLT = st.number_input("PLT (10^9/L):")
Pre_vWF_A2 = st.number_input("Pre-vWF-A2 (pg/mL):")
vWF_A2_D3 = st.number_input("vWF-A2(D3) (pg/mL):")
Operationtime = st.number_input("Operation time (min):")
Anti_coagulation = st.selectbox('Anti-coagulation', ['No', 'Yes'])
Differentiation = st.selectbox('Differentiation', ['poorly',"moderately",'highly'])
Differentiationmap = {'highly': 0, 'moderately': 1, 'poorly': 2}
Differentiation = Differentiationmap[Differentiation]
Anti_coagulation = 1 if Anti_coagulation == 'Yes' else 0

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    XGB = joblib.load("XGB.pkl")
    calibrated = joblib.load("calibrated.pkl")
    scaler = joblib.load("scaler.pkl")
    
    
    # Store inputs into dataframe
    input_numerical = np.array([Age,Anti_coagulation,D_dimer_D3,D_dimer_D5,Differentiation,FDP,Lymphocyte,Operationtime,PLT,Pre_vWF_A2,vWF_A2_D3]).reshape(1, -1)
    feature_names  = ["Age", "Anti_coagulation", "D_dimer_D3","D_dimer_D5","Differentiation","FDP","Lymphocyte","Operationtime","PLT","Pre_vWF_A2","vWF_A2_D3"]
    input_numericalyuan = pd.DataFrame(input_numerical, columns=feature_names)
    input_numerical = pd.DataFrame(input_numerical, columns=feature_names)

    input_numerical[['D_dimer_D5','vWF_A2_D3','D_dimer_D3','Pre_vWF_A2','Age','PLT','FDP','Lymphocyte','Operationtime']] = scaler.transform(input_numerical[['D_dimer_D5','vWF_A2_D3','D_dimer_D3','Pre_vWF_A2','Age','PLT','FDP','Lymphocyte','Operationtime']])


    prediction_proba = calibrated.predict_proba(input_numerical)[:, 1]
    prediction_proba = (prediction_proba * 100).round(2)
    st.markdown("## **Prediction Probabilities (%)**")
    for prob in prediction_proba:
        st.markdown(f"**{prob:.2f}%**")

  
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

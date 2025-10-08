import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components
from sklearn.calibration import CalibratedClassifierCV
st.markdown('<h2 style="font-size:20px;">XGBoost Model for Postoperative Thrombosis</h2>', unsafe_allow_html=True)

if 'age_valid' not in st.session_state:
    st.session_state.age_valid = True
Age = st.number_input("Age (Years):",
    #min_value=18,      # 最小值
    #max_value=85,      # 最大值
    value=18,          # 默认值（可选，默认为 min_value）
    step=1,            
    help="Must be 18-85 years")
if Age < 18 or Age > 85:
    st.error("Value must be between 18 and 85 yesrs.")
D_dimer_D3 = st.number_input("D-dimer on Postoperative Day 3 (μg/mL):",
   #min_value=0.00,      # 最小值
    #max_value=10.00,      # 最大值
    value=0.00,          # 默认值（可选，默认为 min_value）
    step=0.01,            
    help="Must be 0-10 μg/mL")
if D_dimer_D3 > 10.00:
    st.error("Value must be between <10μg/mL.")
D_dimer_D5 = st.number_input("D-dimer on Postoperative Day 5 (μg/mL):",
    #min_value=0.00,      # 最小值
    #max_value=10.00,      # 最大值
    value=0.00,          # 默认值（可选，默认为 min_value）
    step=0.01,           
    help="Must be 0-10 μg/mL")
if D_dimer_D5 > 10.00:
    st.error("Value must be between <10μg/mL.")
FDP = st.number_input("FDP on Postoperative Day 1 (mg/L):",
   #min_value=0.00,      # 最小值
    #max_value=50.00,      # 最大值
    value=0.00,          # 默认值（可选，默认为 min_value）
    step=0.01,           
    help="Must be 0-50 mg/L")
if FDP > 50.00:
    st.error("Value must be between <50mg/L.")
Lymphocyte = st.number_input("Lymphocyte count on Postoperative Day 1 (10^9/L):",
    #min_value=0.00,      # 最小值
    #max_value=4.00,      # 最大值
    value=0.00,          # 默认值（可选，默认为 min_value）
    step=0.01,           
    help="Must be 0.0-4.0 (10^9/L)")
if Lymphocyte > 4.00:
    st.error("Value must be between <4.0(10^9/L).")
PLT = st.number_input("PLT on Postoperative Day 1 (10^9/L):",
    #min_value=0.0,      # 最小值
    #max_value=600.0,      # 最大值
    value=0.0,          # 默认值（可选，默认为 min_value）
    step=0.1,            
    help="Must be 0-600 (10^9/L)")
if PLT > 600.0:
    st.error("Value must be between <600.0(10^9/L).")
Pre_vWF_A2 = st.number_input("Preoperative vWF-A2 (pg/mL):",
    #min_value=0.00,      # 最小值
    #max_value=10000.00,      # 最大值
    value=0.00,          # 默认值（可选，默认为 min_value）
    step=0.01,            
    help="Must be 0-10000 pg/mL")
if Pre_vWF_A2 > 10000.00:
    st.error("Value must be between <10000pg/mL.")
vWF_A2_D3 = st.number_input("vWF-A2 on Postoperative Day 3 (pg/mL):",
    #min_value=0.00,      # 最小值
    #max_value=10000.00,      # 最大值
    value=0.00,          # 默认值（可选，默认为 min_value）
    step=0.01,            
    help="Must be 0-10000 pg/mL")
if vWF_A2_D3 > 10000.00:
    st.error("Value must be between <10000pg/mL.")
Operationtime = st.number_input("Operation Time (min):",
    #min_value=60,      # 最小值
    #max_value=240,      # 最大值
    value=60,          # 默认值（可选，默认为 min_value）
    step=1,            
    help="Must be 60-240 min")
if Operationtime < 60 or Operationtime > 240:
    st.error("Value must be between 60 and 240 min.")
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
        if prob < 1.0:  
            st.markdown(f"**<1%**")
        else:
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

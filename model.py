import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
data = pd.read_csv("E:/wangjing/model/data.csv", encoding='utf-8')
X = data.drop(columns=['VTE'])
y = data['VTE']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=123
)
scaler = StandardScaler()
X_train[['D5Dimer','vWFD3','D3Dimer','PrevWF','Age','PLT','FDP','Lymphocyte','Operationtime']] = scaler.fit_transform(X_train[[
    'D5Dimer','vWFD3','D3Dimer','PrevWF','Age','PLT','FDP','Lymphocyte','Operationtime']])
X_test[['D5Dimer','vWFD3','D3Dimer','PrevWF','Age','PLT','FDP','Lymphocyte','Operationtime']] = scaler.transform(X_test[[
    'D5Dimer','vWFD3','D3Dimer','PrevWF','Age','PLT','FDP','Lymphocyte','Operationtime']])
X_train = X_train.reindex(sorted(X_train.columns), axis=1)
X_test = X_test.reindex(sorted(X_test.columns), axis=1)
XGB = xgb.XGBClassifier(
        learning_rate=0.5803206633634018,
        max_depth=15,
        min_child_weight=0.01775304667697467,
        subsample=0.743445686896757,
       colsample_bytree=0.6118540283803795,
        reg_lambda=6.8149099063095075,
        n_estimators=272,
        scale_pos_weight=4.063464588085775,
               random_state=42)
XGB.fit(X_train, y_train)
calibrated_clf = CalibratedClassifierCV(XGB, method="isotonic", cv=10)
calibrated_clf.fit(X_train, y_train)

import joblib

joblib.dump(calibrated_clf, "XGB.pkl")
joblib.dump(scaler, "scaler.pkl")
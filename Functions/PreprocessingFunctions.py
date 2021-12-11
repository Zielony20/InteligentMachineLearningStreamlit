import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as sc
from sklearn.preprocessing import MinMaxScaler, StandardScaler ,OneHotEncoder, QuantileTransformer, RobustScaler, Normalizer, PowerTransformer

def resizeColumn(my_dataframe, active_coefficient):

    r_min = min(my_dataframe[active_coefficient])
    r_max = max(my_dataframe[active_coefficient])

    t_min = st.text_input('Minimal Value',value=r_min)
    t_max = st.text_input('Maximum Value',value=r_max)
    if st.button("Apply change"):
        #json_widget_saver['apply_scaler_range'] == "1"
        try:
            t_min = float(t_min)
            t_max = float(t_max)
            my_dataframe[active_coefficient] = (my_dataframe[active_coefficient].astype(float) - r_min) / (
                        r_max - r_min) * (t_max - t_min) + t_min
        except ValueError:
            print("wrong values")

    return my_dataframe

def scaleColumn(my_dataframe, active_coefficient):

    scaler = st.slider("", min_value=-10, max_value=10, value=0, step=1)
    if st.button("Apply change"):
        float_value = float(10)
        power_value = float_value ** scaler
        my_dataframe[active_coefficient] = my_dataframe[active_coefficient].astype(float) * power_value
    return my_dataframe

def changeValueInColumn(my_dataframe,active_coefficient):

    values = my_dataframe[active_coefficient].unique()
    print(values)
    h1, h2, h3, h4 = st.columns((1, 1, 1, 1))
    with h1:
        ValueToChange = st.text_input('Value To Change')
    with h2:
        NewValue = st.text_input('New Value')
    with h3:
        DataType = st.radio("Type of data", ('Int64', 'Float64', 'Boolean', 'String'))
    if st.button("Apply"):
        tempType = my_dataframe[active_coefficient].dtype
        # print(tempType)
    my_dataframe[active_coefficient] = my_dataframe[active_coefficient].replace([ValueToChange], NewValue)
    #for i in my_dataframe[active_coefficient]:

    return my_dataframe

def missingValueToChange(my_dataframe,active_coefficient):
    return my_dataframe

def normalizedAll(dataframe,numerical_cols):
    norm = MinMaxScaler().fit(dataframe[numerical_cols])
    return pd.DataFrame( norm.transform(dataframe[numerical_cols]))

def normalized(dataframe,active_coefficient):

    norm = MinMaxScaler().fit(dataframe[[active_coefficient]])
    #dataframe[active_coefficient] = norm.transform(dataframe[active_coefficient])

    return pd.DataFrame( norm.transform(dataframe[[active_coefficient]]))


def standarizationAll(dataframe,numerical_cols):
    train_stand = dataframe[numerical_cols].copy()

    for i in numerical_cols:
        scale = StandardScaler().fit(train_stand[[i]])
        train_stand[i] = scale.transform(train_stand[[i]])
    return train_stand

def standarization(dataframe,active_coefficient):

    scale = StandardScaler().fit(dataframe[[active_coefficient]])
    dataframe[active_coefficient] = scale.transform(dataframe[[active_coefficient]])
    return dataframe[active_coefficient]

def MyQuantileTransformer(dataframe,column,distribution="uniform",n_quantiles=6):

    qt = QuantileTransformer(n_quantiles=n_quantiles, random_state=0, output_distribution=distribution)
    X = dataframe[column].to_numpy().reshape(-1,1)
    result = qt.fit_transform(X)
    dataframe[column] = result
    return dataframe


def MyRobustScaler(dataframe,column):

    X = dataframe[column].to_numpy().reshape(-1,1)
    rs = RobustScaler().fit_transform(X)
    dataframe[column] = rs
    return dataframe

def MyPowerTransformer(dataframe,column,method="yeo-johnson"):

    X = dataframe[column].to_numpy().reshape(-1, 1)
    pt = PowerTransformer(method=method).fit_transform(X)
    dataframe[column] = pt
    return dataframe

def MyNormalizer(dataframe,column):
    return dataframe

def MyStandardScaler(dataframe,column):
    return dataframe

def MyMinMaxScaler(dataframe,column):
    return dataframe


def CreateNewColumn(dataframe,column,column_name="Copy",operation='Duplicate',numeric_object_cols=""):
    aplay=False
    if (operation == 'Duplicate'):
        if st.button("Aplay duplication"):
            dataframe[column_name] = dataframe[column]
            aplay = True
    if (operation == 'Addition'):
        cols = st.multiselect("choose columns to add", numeric_object_cols,[])
        dataframe[column_name] = 0
        if(st.button("Aplay addition")):
            for c in cols:
                dataframe[column_name] = dataframe[column_name] + dataframe[c]
            aplay = True

    if (operation == 'Multiplication'):
        cols = st.multiselect("choose columns to add", numeric_object_cols, [])

        if st.button("Aplay multiplication"):
            dataframe[column_name] = 1
            for c in cols:
                dataframe[column_name] = dataframe[column_name] * dataframe[c]
            aplay = True

    if (operation == 'Raise to power'):
        power = st.slider("Power", min_value=0, max_value=9, value=1, step=1)

        if st.button("Aplay exponentiation"):
            dataframe[column_name] = np.power(dataframe[column],power)
            aplay = True

    return dataframe, aplay

def encoder(dataframe,active_coefficient):
    pass
    #enc = OneHotEncoder().fit()

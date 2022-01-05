import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as sc
from sklearn.preprocessing import MinMaxScaler, StandardScaler ,OneHotEncoder, QuantileTransformer, RobustScaler, Normalizer, PowerTransformer

def getObjectsColumns(my_dataframe):
    s = (my_dataframe.dtypes == 'object')
    object_cols = list(s[s].index)
    return object_cols

def getNumericalColumns(my_dataframe):
    f = (my_dataframe.dtypes == 'float64')
    float_object_cols = list(f[f].index)
    n = (my_dataframe.dtypes == 'int64')
    numeric_object_cols = list(n[n].index) + float_object_cols
    return numeric_object_cols

def getClassificationColums(my_dataframe : pd.DataFrame):
    u = list(my_dataframe.nunique())
    class_object_cols = []
    for i in range(len(u)):
        if u[i]<=6:
            class_object_cols.append(my_dataframe.columns[i])
        pass
    return class_object_cols

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
    h1, h2, h3, h4 = st.columns((1, 1, 1, 1))
    with h1:
        ValueToChange = st.text_input('Value To Change')
    with h2:
        NewValue = st.text_input('New Value')
    with h3:
        DataType = st.radio("Type of data", ('Int64', 'Float64', 'Boolean', 'String'))
    if st.button("Apply"):
        tempType = my_dataframe[active_coefficient].dtype
    my_dataframe[active_coefficient] = my_dataframe[active_coefficient].replace([ValueToChange], NewValue)
    #for i in my_dataframe[active_coefficient]:

    return my_dataframe

def missingValueToChange(my_dataframe : pd.DataFrame,active_coefficient,strategy):
    #import numbers
    nanList = ["<Nan>","<nan>"]
    #my_dataframe = CreateNewColumn(my_dataframe, active_coefficient, column_name="Copy", operation='Duplicate')
    #my_dataframe,_,_ = dropColumn(my_dataframe, active_coefficient)
    #x = lambda a : np.nan if isinstance(a, numbers.Number) else np.nan
    #my_dataframe.insert(0,active_coefficient,[ x(i) for i in my_dataframe["Copy"] ],True)
    #my_dataframe, _, _ = dropColumn(my_dataframe, "Copy")
    if strategy == "median":
        median = my_dataframe[active_coefficient].median()
        my_dataframe[active_coefficient] = my_dataframe[active_coefficient].fillna(0, inplace=True)
    if strategy == "average":
        average = my_dataframe[active_coefficient].average()
        my_dataframe[active_coefficient] = my_dataframe[active_coefficient].fillna(average, inplace=True)
    return my_dataframe

def dropColumn(my_dataframe : pd.DataFrame,active_coefficient,float_object_cols=None):
    my_dataframe = my_dataframe.drop(active_coefficient, axis=1)
    numeric = list(float_object_cols)
    if (active_coefficient in numeric):
        numeric.remove(active_coefficient)
    active_coefficient = str(numeric[0])
    return my_dataframe, active_coefficient, numeric

def dropRows(my_dataframe,active_coefficient):
    my_dataframe = my_dataframe.dropna(axis=0, how='any', thresh=None, subset=[active_coefficient], inplace=False)
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

#def MyNormalizer(dataframe,column):
#    return dataframe

#def MyStandardScaler(dataframe,column):
#    return dataframe

#def MyMinMaxScaler(dataframe,column):
#    return dataframe


def CreateNewColumn(dataframe,column,column_name="Copy",operation='Duplicate',numeric_object_cols=""):
    apply=False
    if (operation == 'Duplicate'):
        if st.button("Apply duplication"):
            dataframe[column_name] = dataframe[column]
            Apply = True
    if (operation == 'Addition'):
        cols = st.multiselect("choose columns to add", numeric_object_cols,[])
        dataframe[column_name] = 0
        if(st.button("Apply addition")):
            for c in cols:
                dataframe[column_name] = dataframe[column_name] + dataframe[c]
            apply = True

    if (operation == 'Multiplication'):
        cols = st.multiselect("choose columns to add", numeric_object_cols, [])

        if st.button("Apply multiplication"):
            dataframe[column_name] = 1
            for c in cols:
                dataframe[column_name] = dataframe[column_name] * dataframe[c]
            apply = True

    if (operation == 'Raise to power'):
        power = st.slider("Power", min_value=0, max_value=9, value=1, step=1)

        if st.button("Apply exponentiation"):
            dataframe[column_name] = np.power(dataframe[column],power)
            apply = True

    return dataframe, apply

def encoder(dataframe,active_coefficient):
    pass
    #enc = OneHotEncoder().fit()


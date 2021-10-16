import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split


def loadInterface():
    my_dataframe = pd.read_csv(PWD + '/data.csv')
    dataFrameWidget = st.dataframe(my_dataframe)

    #print("Variables types")
    #print(my_dataframe.dtypes)

    s = (my_dataframe.dtypes == 'object')
    object_cols = list(s[s].index)

    #print("Categorical variables:")
    #print(object_cols)

    n = (my_dataframe.dtypes == 'int64')
    numeric_object_cols = list(n[n].index)

    #print("Numerical Variable")
    #print(numeric_object_cols)

    f = (my_dataframe.dtypes == 'float64')
    float_object_cols = list(f[f].index)

    #print("Float Variable")
    #print(float_object_cols)

    # cols_with_missing = [col for col in my_dataframe.columns if my_dataframe[col].isnull().any()]
    # my_dataframe.drop(cols_with_missing, axis=1, inplace=True)
    column_names = my_dataframe.columns
    #print(column_names)
    variable_numbers = len(my_dataframe.columns)
    active_coefficient = json_widget_saver['active_coefficient']

    with st.sidebar:
        csv = convert_df(my_dataframe)
        st.download_button(
            label="Download data as CSV ",
            data=csv,
            file_name='my_df.csv',
            mime='text/csv', )


        buttons = list()
        buttons = st.columns(variable_numbers)
        index = 0

        active_coefficient = st.selectbox(
            'Which coefficient would you like to analizing?',
            column_names)
        json_widget_saver['active_coefficient'] = active_coefficient
        saveWidgets()

        value_to_predict = st.selectbox(
            'Which coefficient would you like to predict?',
            column_names)
        #json_widget_saver['value_to_predict'] = value_to_predict
        #saveWidgets()
        
    column_number = len(my_dataframe)

    # train_set = my_dataframe[0:column_number*0.8]
    # test_set = my_dataframe[column_number*0.8+1:column_number*0.8]

    X_train, X_valid, y_train, y_valid = train_test_split(my_dataframe[0:column_number], range(column_number),
                                                          train_size=0.8, test_size=0.2, random_state=42)

    if active_coefficient != "":
        st.title("Active: "+active_coefficient)
        btn1, btn2, btn3, btn4 = st.columns((1, 1, 1, 1))
        with btn1:
            if st.button("Change Value"):
                json_widget_saver['change_value_btn'] = "1"
                json_widget_saver['scale_btn'] = ""
                json_widget_saver['missing_value_btn'] = ""
        with btn2:
            if st.button("Scale"):
                json_widget_saver['change_value_btn'] = ""
                json_widget_saver['scale_btn'] = "1"
                json_widget_saver['missing_value_btn'] = ""
        with btn3:
            if st.button("Missing Value Strategy"):
                json_widget_saver['change_value_btn'] = ""
                json_widget_saver['scale_btn'] = ""
                json_widget_saver['missing_value_btn'] = "1"

        if json_widget_saver['change_value_btn'] == "1":
            h1, h2, h3, h4 = st.columns((1, 1, 1, 1))
            with h1:
                ValueToChange = st.text_input('Value To Change')
            with h2:
                NewValue = st.text_input('New Value')
            with h3:
                DataType = st.radio("Type of data",('Int64', 'Float64', 'Boolean', 'String'))

            if st.button("Apply"):
                tempType = my_dataframe[active_coefficient].dtype
                #print(tempType)
                my_dataframe[active_coefficient] = changeValueInColumn(my_dataframe,active_coefficient,ValueToChange,NewValue,DataType)
                dataFrameWidget.dataframe(my_dataframe)

        if json_widget_saver['scale_btn'] == "1":
            scaler = st.slider("", min_value=-10, max_value=10, value=0, step=1)
            if st.button("Apply change"):
                float_value = float(10)
                power_value = float_value ** scaler
                my_dataframe[active_coefficient] = my_dataframe[active_coefficient].astype(float) * power_value
                #print(my_dataframe[active_coefficient])
                st.title(power_value)
                dataFrameWidget.empty()
                dataFrameWidget.dataframe(my_dataframe)


        # Space out the maps so the first one is 2x the size of the other three
        c1, c2, c3, c4 = st.columns((1, 1, 1, 1))
        columns_array = [c1, c2, c3, c4]
        counter = 0
        for coefficient in numeric_object_cols:

            if counter == 1:
                with c1:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 2:
                with c2:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 3:
                with c3:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 4:
                with c4:
                    print_chart(my_dataframe, active_coefficient, coefficient)

            counter += 1
            if counter >= 5:
                counter = 1


        for coefficient in float_object_cols:
            if counter == 1:
                with c1:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 2:
                with c2:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 3:
                with c3:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 4:
                with c4:
                    print_chart(my_dataframe, active_coefficient, coefficient)

            counter += 1
            if counter >= 5:
                counter = 1

    #download_csv(my_dataframe)
    # Use the full page instead of a narrow central column

    csv = convert_df(my_dataframe)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='my_df.csv',
        mime='text/csv', )


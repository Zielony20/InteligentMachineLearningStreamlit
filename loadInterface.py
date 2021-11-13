import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from sklearn import datasets

def loadInterface():

    my_dataframe = pd.read_csv(PWD + '/data.csv',index_col=False)
    #if 'Unnamed: 0' in my_dataframe.columns:
    #    my_dataframe.drop(['Unnamed: 0'], axis=1)
    dataFrameWidget = st.dataframe(my_dataframe)

    s = (my_dataframe.dtypes == 'object')
    object_cols = list(s[s].index)
    f = (my_dataframe.dtypes == 'float64')
    float_object_cols = list(f[f].index)
    n = (my_dataframe.dtypes == 'int64')
    numeric_object_cols = list(n[n].index) + float_object_cols
    column_names = my_dataframe.columns
    variable_numbers = len(my_dataframe.columns)
    column_number = len(my_dataframe)
    active_coefficient = json_widget_saver['active_coefficient']

####################################################
#####################sidebar########################
####################################################

    with st.sidebar:
        csv = convert_df(my_dataframe)
        st.download_button(
            label="Download data as CSV ",
            data=csv,
            file_name='my_df.csv',
            mime='text/csv', )

        active_coefficient = st.selectbox(
            'Which coefficient would you like to analizing?',
            column_names)
        json_widget_saver['active_coefficient'] = active_coefficient
        saveWidgets()

        if active_coefficient != "":

            preprocessing = st.selectbox(
                'Which coefficient would you like to analizing?',
                ['Change Value', 'Scale','Resize Range', 'Missing Value Strategy', 'Normalization', 'Standarization'])

            savePreprocesingButtons(preprocessing)

            if (preprocessing == 'Normalization'):
                if st.button("Aplay Normalization"):
                    my_dataframe[numeric_object_cols] = normalized(my_dataframe, numerical_cols=numeric_object_cols)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
            if (preprocessing == 'Standarization'):
                if st.button("Aplay Standarization"):
                    my_dataframe[numeric_object_cols] = standarization(dataframe=my_dataframe,
                                                                       numerical_cols=numeric_object_cols)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()

            if json_widget_saver['change_value_btn'] == "1":

                my_dataframe = changeValueInColumn(my_dataframe, active_coefficient)
                dataFrameWidget = refreshDataFrameWidget(dataFrameWidget, my_dataframe)
            if json_widget_saver['scale_btn'] == "1":

                my_dataframe = scaleColumn(my_dataframe, active_coefficient)
                dataFrameWidget = refreshDataFrameWidget(dataFrameWidget,my_dataframe)

            if json_widget_saver['resize_range_btn'] == "1":

                my_dataframe = resizeColumn(my_dataframe,active_coefficient)
                dataFrameWidget = refreshDataFrameWidget(dataFrameWidget,my_dataframe)



########################################################
#End of sidebar

    _, histPlace, _ = st.columns((1, 1, 1))
    with histPlace:
        fig, ax = plt.subplots(edgecolor = 'black')
        ax.hist(my_dataframe[active_coefficient].to_numpy(), bins=20, edgecolor = 'black')
        ax.set_facecolor("gray")

        st.pyplot(fig)
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

    #download_csv(my_dataframe)
    # Use the full page instead of a narrow central column

    csv = convert_df(my_dataframe)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='my_df.csv',
        mime='text/csv', )

    value_to_predict = st.selectbox(
        'Which coefficient would you like to predict?',
        numeric_object_cols)
    json_widget_saver['value_to_predict'] = value_to_predict
    saveWidgets()

    optionUseToPredict = [i for i in numeric_object_cols]
    optionUseToPredict.remove(value_to_predict)

    option_use_to_predict = st.multiselect(
        'Which coefficient use to predict',
        [i for i in numeric_object_cols],
        optionUseToPredict)

    algorithm_model = st.selectbox(
        'Which Machine Learning model use?',
        ['LinearRegression', 'RandomForestRegressor', 'KNeighborsClassifier', 'GaussianNB', 'KMeans'])

    if st.button("Create Model"):
        createModel(my_dataframe, option_use_to_predict, value_to_predict, algorithm_model)
    if st.button("Test Model"):
        testModel()




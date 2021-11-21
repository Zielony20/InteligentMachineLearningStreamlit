import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
from functions import *
from sklearn import datasets
import plotly.figure_factory as ff
import plotly.express as px

def loadInterface():

    my_dataframe = pd.read_csv(PWD + '/data.csv',index_col=None)
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
            numeric_object_cols)
        json_widget_saver['active_coefficient'] = active_coefficient
        saveWidgets()

        if active_coefficient != "":

            preprocessing = st.selectbox(
                'Choose preprocessing operation',
                ['Change Value', 'Scale','Resize Range', 'Missing Value Strategy', 'Normalization', 'Standarization'])

            savePreprocesingButtons(preprocessing)

            if (preprocessing == 'Normalization'):
                if st.button("Aplay normalization"):
                    my_dataframe[active_coefficient] = normalized(my_dataframe, active_coefficient)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()

                if st.button("Aplay normalization on all columns"):
                    my_dataframe[numeric_object_cols] = normalizedAll(my_dataframe, numerical_cols=numeric_object_cols)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()

            if (preprocessing == 'Standarization'):
                if st.button("Aplay standarization"):
                    my_dataframe[active_coefficient] = standarization(dataframe=my_dataframe, active_coefficient=active_coefficient)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
                if st.button("Aplay standarization on all columns"):
                    my_dataframe[numeric_object_cols] = standarizationAll(dataframe=my_dataframe,
                                                                       numerical_cols=numeric_object_cols)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()


            if json_widget_saver['missing_value_btn']:
                if st.button("Delete column"):
                    if st.button("Are sure? This operation cannot be undone"):
                        my_dataframe.dropna(subset=[active_coefficient])
                        json_widget_saver['']
                        dataFrameWidget.empty()
                        dataFrameWidget.dataframe(my_dataframe)
                        my_dataframe.to_csv(PWD + '/data.csv', index=False)
                        saveWidgets()
                if st.button("Delete rows"):
                    if st.button("Are sure? This operation cannot be undone"):
                        my_dataframe.drop(active_coefficient, axis=1)
                        dataFrameWidget.empty()
                        dataFrameWidget.dataframe(my_dataframe)
                        my_dataframe.to_csv(PWD + '/data.csv', index=False)
                        saveWidgets()
                if st.button("Replace missing value with..."):
                    st.title("NotImplemented")

            if json_widget_saver['change_value_btn'] == "1":
                pass
            #    my_dataframe = changeValueInColumn(my_dataframe, active_coefficient)
             #   dataFrameWidget = refreshDataFrameWidget(dataFrameWidget, my_dataframe)
            if json_widget_saver['scale_btn'] == "1":

                my_dataframe = scaleColumn(my_dataframe, active_coefficient)
                dataFrameWidget = refreshDataFrameWidget(dataFrameWidget,my_dataframe)

            if json_widget_saver['resize_range_btn'] == "1":

                my_dataframe = resizeColumn(my_dataframe,active_coefficient)
                dataFrameWidget = refreshDataFrameWidget(dataFrameWidget,my_dataframe)



########################################################
#End of sidebar

    kolmogorov, histPlace, _ = st.columns((1, 4, 1))
    with histPlace:
        int_val = [.01] #st.number_input('hist bins', value=1, step=1,format="%.2f")
        group_labels = [active_coefficient]
        hist_data = [my_dataframe[active_coefficient].to_numpy()]
        # Create distplot with custom bin_size
#        fig = ff.create_distplot(
 #           hist_data, group_labels, bin_size=int_val, histnorm="probability density")
        # Plot!
  #      st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(my_dataframe[active_coefficient], x=active_coefficient,facet_col_spacing=1,marginal="violin",histnorm=None,barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

        with kolmogorov:

            st.text('\n')
            st.text('\n')
            st.text('\n')
            histSimilarity(my_dataframe[active_coefficient].to_numpy())

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
        ['LinearRegression', 'DecisionTree','RandomForestRegressor', 'KNeighborsClassifier', 'GaussianNB', 'KMeans'])
    metrics = st.multiselect(
        "Which metrics show?",
        ['MAE', 'MSE', 'RMSE', 'RMSLE', 'R squared'],
        []
    )
    if st.button("Create Model"):
        createModel(my_dataframe, option_use_to_predict, value_to_predict, algorithm_model, metrics)




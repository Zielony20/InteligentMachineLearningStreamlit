from Functions.ModelFunctions import *
from Functions.FileSystemFunctions import *
from Functions.JsonHandler import *
from Functions.ChartsFunctions import *
from Functions.PreprocessingFunctions import *

def loadInterface():

    my_dataframe = pd.read_csv(PWD + '/data.csv',index_col=None)
    dataFrameWidget = st.dataframe(my_dataframe)
    object_cols = getObjectsColumns(my_dataframe)
    numeric_object_cols = getNumericalColumns(my_dataframe)
    active_coefficient = json_widget_saver['active_coefficient']

    class_object_cols = getClassificationColums(my_dataframe)

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
            my_dataframe.columns)
        json_widget_saver['active_coefficient'] = active_coefficient
        if active_coefficient != "":

            if(active_coefficient in numeric_object_cols):
                preprocessing = st.selectbox(
                'Choose preprocessing operation',
                ['Rename', 'Create New Column', 'Delete column', 'Scale', 'Resize Range', 'Missing Value Strategy',
                 'Normalization', 'Standarization', 'Quantile Transformer',
                 'Robust Scaler', 'Power Transformer'])
            else:
                preprocessing = st.selectbox(
                    'Choose preprocessing operation',
                    ['Rename','Create New Column','Delete column','Missing Value Strategy'])

            if (preprocessing == 'Rename'):
                column_name = st.text_input(label="New name",value=active_coefficient)
                if st.button("Apply"):
                    my_dataframe.rename(columns={active_coefficient: column_name}, inplace=True)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    active_coefficient = column_name
                    saveWidgets()
                    st.experimental_rerun()
            if (preprocessing == 'Create New Column'):

                column_name = st.text_input("column name")
                operation = st.selectbox('choose operation',
                                         ['Duplicate','Addition','Multiplication','Raise to power'])
                apply=False
                my_dataframe,apply = CreateNewColumn(my_dataframe,active_coefficient,column_name,operation,numeric_object_cols)
                if apply:
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
                    st.experimental_rerun()
            if (preprocessing == 'Power Transformer'):
                method = st.selectbox('method',['yeo-johnson','box-cox'])
                if (st.button("Apply Power Transformer")):
                    my_dataframe = MyPowerTransformer(my_dataframe, active_coefficient, method)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
            if (preprocessing == 'Robust Scaler'):
                if(st.button("Apply Robust Scaler")):
                    my_dataframe = MyRobustScaler(my_dataframe, active_coefficient)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
            if (preprocessing == 'Quantile Transformer'):
                distribution = st.selectbox('distribution',
                                            ['uniform','normal'])
                n_quantiles = st.slider("n_quantiles", min_value=0, max_value=100, value=5, step=1)
                if (st.button("Apply Quantile Transformer" )):
                    my_dataframe = MyQuantileTransformer(my_dataframe,active_coefficient,distribution,n_quantiles)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
            if (preprocessing == 'Normalization'):
                if st.button("Apply normalization"):
                    my_dataframe[active_coefficient] = normalized(my_dataframe, active_coefficient)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()

                if st.button("Apply normalization on all columns"):
                    my_dataframe[numeric_object_cols] = normalizedAll(my_dataframe, numerical_cols=numeric_object_cols)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()

            if (preprocessing == 'Standarization'):
                if st.button("Apply standarization"):
                    my_dataframe[active_coefficient] = standarization(dataframe=my_dataframe, active_coefficient=active_coefficient)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
                if st.button("Apply standarization on all columns"):
                    my_dataframe[numeric_object_cols] = standarizationAll(dataframe=my_dataframe,
                                                                       numerical_cols=numeric_object_cols)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()


#            'Scale', 'Resize Range', 'Missing Value Strategy'
            if preprocessing == 'Delete column':
                if st.button("Delete column"):
                    my_dataframe, active_coefficient, numeric_object_cols = dropColumn(my_dataframe,active_coefficient,numeric_object_cols)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    json_widget_saver['active_coefficient'] = active_coefficient
                    saveWidgets()
                    st.experimental_rerun()

            if preprocessing == 'Missing Value Strategy':
                if st.button("Delete column"):
                    my_dataframe, active_coefficient, numeric_object_cols = dropColumn(my_dataframe,active_coefficient,numeric_object_cols)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    json_widget_saver['active_coefficient'] = active_coefficient
                    saveWidgets()
                    st.experimental_rerun()

                if st.button("Delete rows"):
                    my_dataframe = dropRows(my_dataframe,active_coefficient)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
                if st.button("Replace missing value with median"):
                    my_dataframe = missingValueToChange(my_dataframe, active_coefficient, "median")
                    print(my_dataframe[active_coefficient])
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
                    st.experimental_rerun()
                if st.button("Replace missing value with average"):
                    my_dataframe = missingValueToChange(my_dataframe, active_coefficient, "average")
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
                    st.experimental_rerun()

            if preprocessing == 'Scale':
                my_dataframe = scaleColumn(my_dataframe, active_coefficient)
                #dataFrameWidget = refreshDataFrameWidget(dataFrameWidget, my_dataframe)
                dataFrameWidget.empty()
                dataFrameWidget.dataframe(my_dataframe)
                my_dataframe.to_csv(PWD + '/data.csv', index=False)
                saveWidgets()
            if preprocessing == 'Resize Range':
                my_dataframe = resizeColumn(my_dataframe, active_coefficient)
                dataFrameWidget.empty()
                dataFrameWidget.dataframe(my_dataframe)
                my_dataframe.to_csv(PWD + '/data.csv', index=False)
                saveWidgets()

#End of sidebar

    if active_coefficient in numeric_object_cols:

        _,title,_ = st.columns((1,1,1))
        with title:
            #st.title(active_coefficient)
            st.markdown("<h1 style='text-align: center; color: Black;'> %s </h1>"%active_coefficient, unsafe_allow_html=True)
        if active_coefficient in class_object_cols:
            counterPieChart(my_dataframe,active_coefficient)
        else:
            histogramWithKomogorov(active_coefficient,my_dataframe)

        #comparisonCharts(active_coefficient,my_dataframe,numeric_object_cols)
        chartsCoordinator(active_coefficient, my_dataframe, numeric_object_cols)
        targets = pf.getClassificationColums(my_dataframe)
        if (len(numeric_object_cols) <= 6 and (len(targets) > 0)):
            target = st.selectbox(
                    'Target ',
                    targets)
            crossCharts(my_dataframe, target)



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

    if value_to_predict in class_object_cols:

        algorithm_model = st.selectbox(
        'Which Machine Learning model use?',
        ['DecisionTree','RandomForestRegressor', 'KNeighborsClassifier','LogisticRegression','SGDClassifier'])

    else:
        algorithm_model = st.selectbox(
            'Which Machine Learning model use?',
            ['LinearRegression', 'DecisionTree', 'RandomForestRegressor'])

    metrics = st.multiselect(
        "Which metrics show?",
        ['MAE', 'MSE', 'RMSE', 'RMSLE', 'R squared'],
        []
    )

    createModel(my_dataframe, option_use_to_predict, value_to_predict, algorithm_model, metrics)



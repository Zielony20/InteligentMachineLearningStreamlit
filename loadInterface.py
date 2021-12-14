from Functions.ModelFunctions import *
from Functions.FilesystemFunctions import *
from Functions.JsonHandler import *
from Functions.ChartsFunctions import *
from Functions.PreprocessingFunctions import *
import time
def loadInterface():

    my_dataframe = pd.read_csv(PWD + '/data.csv',index_col=None)
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
                ['Create New Column' ,'Scale','Resize Range','Delete column', 'Missing Value Strategy',
                 'Normalization', 'Standarization', 'Quantile Transformer',
                 'Robust Scaler', 'Power Transformer'])

            

            if (preprocessing == 'Create New Column'):

                column_name = st.text_input("column name")
                operation = st.selectbox('choose operation',
                                         ['Duplicate','Addition','Multiplication','Raise to power'])
                aplay=False
                my_dataframe,aplay = CreateNewColumn(my_dataframe,active_coefficient,column_name,operation,numeric_object_cols)
                if aplay:
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
            if (preprocessing == 'Power Transformer'):
                method = st.selectbox('method',['yeo-johnson','box-cox'])
                if (st.button("Aplay Power Transformer")):
                    my_dataframe = MyPowerTransformer(my_dataframe, active_coefficient, method)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
            if (preprocessing == 'Robust Scaler'):
                if(st.button("Aplay Robust Scaler")):
                    my_dataframe = MyRobustScaler(my_dataframe, active_coefficient)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
            if (preprocessing == 'Quantile Transformer'):
                distribution = st.selectbox('distribution',
                                            ['uniform','normal'])
                n_quantiles = st.slider("n_quantiles", min_value=0, max_value=100, value=5, step=1)
                if (st.button("Aplay Quantile Transformer" )):
                    my_dataframe = MyQuantileTransformer(my_dataframe,active_coefficient,distribution,n_quantiles)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
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


#            'Scale', 'Resize Range', 'Missing Value Strategy'
            if preprocessing == 'Delete column':
                if st.button("Delete column"):
                    my_dataframe, active_coefficient, numeric_object_cols = dropColumn(my_dataframe,active_coefficient,numeric_object_cols)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    json_widget_saver['active_coefficient'] = active_coefficient
                    saveWidgets()

            if preprocessing == 'Missing Value Strategy':
                if st.button("Delete column"):
                    my_dataframe, active_coefficient, numeric_object_cols = dropColumn(my_dataframe,active_coefficient,numeric_object_cols)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    json_widget_saver['active_coefficient'] = active_coefficient
                    saveWidgets()


                if st.button("Delete rows"):
                    my_dataframe = dropRows(my_dataframe,active_coefficient)
                    dataFrameWidget.empty()
                    dataFrameWidget.dataframe(my_dataframe)
                    my_dataframe.to_csv(PWD + '/data.csv', index=False)
                    saveWidgets()
                if st.button("Replace missing value with..."):
                    if st.button("median"):
                        my_dataframe = missingValueToChange(my_dataframe, active_coefficient, "median")
                        dataFrameWidget.empty()
                        dataFrameWidget.dataframe(my_dataframe)
                        my_dataframe.to_csv(PWD + '/data.csv', index=False)
                        saveWidgets()
                    if st.button("average"):
                        my_dataframe = missingValueToChange(my_dataframe, active_coefficient, "average")
                        dataFrameWidget.empty()
                        dataFrameWidget.dataframe(my_dataframe)
                        my_dataframe.to_csv(PWD + '/data.csv', index=False)
                        saveWidgets()

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
    _,title,_ = st.columns((1,1,1))
    with title:
        #st.title(active_coefficient)
        st.markdown("<h1 style='text-align: center; color: White;'> %s </h1>"%active_coefficient, unsafe_allow_html=True)
    histogramWithKomogorov(active_coefficient,my_dataframe)
    comparisonCharts(active_coefficient,my_dataframe,numeric_object_cols)

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
        ['LinearRegression', 'DecisionTree','RandomForestRegressor', 'KNeighborsClassifier', 'GaussianNB'])
    metrics = st.multiselect(
        "Which metrics show?",
        ['MAE', 'MSE', 'RMSE', 'RMSLE', 'R squared'],
        []
    )

    createModel(my_dataframe, option_use_to_predict, value_to_predict, algorithm_model, metrics)



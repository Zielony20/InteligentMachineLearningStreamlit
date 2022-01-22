from Functions.ModelFunctions import *
from Functions.FileSystemFunctions import *
from Functions.JsonHandler import *
from Functions.ChartsFunctions import *
from Functions.PreprocessingFunctions import *

def loadInterface():
    original_dataframe = pd.read_csv(PWD + '/original.csv',index_col=None)
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
        if st.button("Load original data"):
            my_dataframe = loadCsv("original.csv")
                #pd.read_csv(PWD + '/original.csv', index_col=None)
            saveAll(dataFrameWidget, my_dataframe, rerun=True)

        if st.button("Undo last change"):
            my_dataframe = loadCsv('lastchange.csv')
               # pd.read_csv(PWD + '/lastchange.csv', index_col=None)
            #my_dataframe, active_coefficient, numeric_object_cols = dropColumn(my_dataframe, active_coefficient,
            #                                                                   numeric_object_cols)
            saveAll(dataFrameWidget, my_dataframe, rerun=True)


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
                ['Rename', 'Create New Column','Math Operation' , 'Delete column', 'Scale', 'Resize Range',
                 'Normalization', 'Standarization', 'Quantile Transformer',
                 'Robust Scaler', 'Power Transformer'])
                # inactive
                #,'Missing Value Strategy', 'PolynomialTransform'
            else:
                preprocessing = st.selectbox(
                    'Choose preprocessing operation',
                    ['Rename','Create New Column','Delete column']) #,'Missing Value Strategy'

            if (preprocessing == 'Rename'):
                column_name = st.text_input(label="New name",value=active_coefficient)
                if st.button("Apply"):
                    my_dataframe.rename(columns={active_coefficient: column_name}, inplace=True)
                    dataFrameWidget = saveAll(dataFrameWidget, my_dataframe ,rerun=True, active_coefficient=True)
            if (preprocessing == 'Math Operation'):

                my_dataframe, apply = mathOperation(my_dataframe, active_coefficient)
                if apply:
                    saveAll(dataFrameWidget, my_dataframe, rerun=True)

                apply = False

            if (preprocessing == 'Create New Column'):

                column_name = st.text_input("column name")
                operation = st.selectbox('choose operation',
                                         ['Duplicate','Addition','Multiplication','Raise to power'])
                apply = False
                my_dataframe,apply = CreateNewColumn(my_dataframe,active_coefficient,column_name,operation,numeric_object_cols)
                if apply:
                    saveAll(dataFrameWidget, my_dataframe ,rerun=True)

            if (preprocessing == 'Power Transformer'):

                columns = st.multiselect(
                    "Which columns modify?",
                    numeric_object_cols,
                    [active_coefficient]
                )
                moreThanZero = True
                for i in columns:
                    if min(my_dataframe[i]) <= 0:
                        moreThanZero = False
                        break

                if (moreThanZero):
                    methods = ['yeo-johnson', 'box-cox']
                else:
                    methods = ['yeo-johnson']
                method = st.selectbox('method', methods)

                if (st.button("Apply Power Transformer")):
                    my_dataframe = MyPowerTransformer(my_dataframe, columns, method)
                    saveAll(dataFrameWidget, my_dataframe ,rerun=True)

            if (preprocessing == 'Robust Scaler'):
                columns = st.multiselect(
                    "Which columns modify?",
                    numeric_object_cols,
                    [active_coefficient]
                )
                if(st.button("Apply Robust Scaler")):
                    my_dataframe = MyRobustScaler(my_dataframe, columns)
                    saveAll(dataFrameWidget, my_dataframe ,rerun=False)

            if (preprocessing == 'Quantile Transformer'):
                columns = st.multiselect(
                    "Which columns modify?",
                    numeric_object_cols,
                    [active_coefficient]
                )
                distribution = st.selectbox('distribution',
                                            ['uniform','normal'])
                n_quantiles = st.slider("n_quantiles", min_value=0, max_value=100, value=5, step=1)
                if (st.button("Apply Quantile Transformer" )):
                    my_dataframe = MyQuantileTransformer(my_dataframe,columns,distribution,n_quantiles)
                    saveAll(dataFrameWidget, my_dataframe ,rerun=False)

            if (preprocessing == 'Normalization'):
                if st.button("Apply normalization"):
                    my_dataframe[active_coefficient] = normalized(my_dataframe, active_coefficient)
                    saveAll(dataFrameWidget, my_dataframe ,rerun=False)

                if st.button("Apply normalization on all columns"):
                    my_dataframe[numeric_object_cols] = normalizedAll(my_dataframe, numerical_cols=numeric_object_cols)
                    saveAll(dataFrameWidget, my_dataframe ,rerun=False)

            if (preprocessing == 'Standarization'):
                if st.button("Apply standarization"):
                    my_dataframe[active_coefficient] = standarization(dataframe=my_dataframe, active_coefficient=active_coefficient)
                    saveAll(dataFrameWidget, my_dataframe ,rerun=False)
                if st.button("Apply standarization on all columns"):
                    my_dataframe[numeric_object_cols] = standarizationAll(dataframe=my_dataframe,
                                                                       numerical_cols=numeric_object_cols)
                    saveAll(dataFrameWidget, my_dataframe ,rerun=False)

            if preprocessing == 'Delete column':
                if st.button("Delete column"):
                    my_dataframe, active_coefficient, numeric_object_cols = dropColumn(my_dataframe,active_coefficient,numeric_object_cols)
                    saveAll(dataFrameWidget, my_dataframe, rerun=True, active_coefficient=True)

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
                    #print(my_dataframe[active_coefficient])
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
            if preprocessing == 'PolynomialTransform':

                degree = st.slider("degree", min_value=0, max_value=360, value=0, step=1)
                if st.button("transform"):
                    my_dataframe = polyFeature(my_dataframe, active_coefficient, degree)
                    saveAll(dataFrameWidget, my_dataframe, rerun=True, active_coefficient=True)

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


        charts(my_dataframe, active_coefficient)

    correlationHeatmap(my_dataframe)
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
        ['LinearRegression','DecisionTreeClassifier','RandomForestRegressor', 'KNeighborsClassifier','LogisticRegression','SGDClassifier'])

    else:
        algorithm_model = st.selectbox(
            'Which Machine Learning model use?',
            ['LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor','Lasso','SupportVectorRegression'])

    metrics = st.multiselect(
        "Which metrics show?",
        ['classification score','MAE', 'MSE', 'RMSE', 'RMSLE'],
        ['classification score','MAE', 'MSE', 'RMSE']
    )
    original_option_use_to_predict = list()
    for i in original_dataframe.columns:
        if i in option_use_to_predict:
            original_option_use_to_predict.append(i)
    readyToTests ,model = createModel([my_dataframe,original_dataframe], option_use_to_predict, value_to_predict, algorithm_model)

    if(readyToTests):
        modify_data, original_data = st.columns((1,1))
        with modify_data:
            st.title("Data with preprocessing")
            trainX, validX, trainY, validY, X, Y = splitData(my_dataframe,option_use_to_predict,value_to_predict)
            testModel(model, trainX, validX, trainY, validY, X, Y , metrics)
        with original_data:
            st.title("Original data")
            trainX, validX, trainY, validY, X, Y = splitData(original_dataframe,original_option_use_to_predict,value_to_predict)
            testModel(model, trainX, validX, trainY, validY, X, Y , metrics)





from Functions.ModelFunctions import *
from Functions.FileSystemFunctions import *
from Functions.JsonHandler import *
from Functions.ChartsFunctions import *
from Functions.PreprocessingFunctions import *
import joblib


def loadInterface():
    original_dataframe = pd.read_csv(PWD + '/original.csv',index_col=None)
    my_dataframe = pd.read_csv(PWD + '/data.csv',index_col=None)
    dataFrameWidget = st.dataframe(my_dataframe)
    object_cols = getObjectsColumns(my_dataframe)
    numeric_object_cols = getNumericalColumns(my_dataframe)
    active_coefficient = json_widget_saver['active_coefficient']
    class_object_cols = getClassificationColums(my_dataframe)


#####################sidebar########################

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
                columns = st.multiselect(
                    "Which columns modify?",
                    numeric_object_cols,
                    [active_coefficient]
                )
                if st.button("Apply normalization"):

                    my_dataframe[numeric_object_cols] = normalizedAll(my_dataframe, columns)
                    saveAll(dataFrameWidget, my_dataframe ,rerun=False)

            if (preprocessing == 'Standarization'):
                columns = st.multiselect(
                    "Which columns modify?",
                    numeric_object_cols,
                    [active_coefficient]
                )
                if st.button("Apply standarization"):

                    my_dataframe[numeric_object_cols] = standarizationAll(my_dataframe,columns)
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

#################End of sidebar#####################

    if active_coefficient in numeric_object_cols:

        _,title,_ = st.columns((1,1,1))
        with title:
            pass
            #st.title(active_coefficient)
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
        ['DecisionTreeClassifier','RandomForestRegressor', 'KNeighborsClassifier','LogisticRegression','SGDClassifier'])

    else:
        algorithm_model = st.selectbox(
            'Which Machine Learning model use?',
            ['LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor','Lasso','SupportVectorRegression', "KNeighborsRegressor"])

    metrics = st.multiselect(
        "Which metrics show?",
        ['classification score','MAE', 'MSE', 'RMSE', 'RMSLE'],
        ['MAE', 'MSE', 'RMSE']
    )

    original_option_use_to_predict = list()
    for i in original_dataframe.columns:
        if i in option_use_to_predict:
            original_option_use_to_predict.append(i)

    readyToTests ,model = createModel(algorithm_model)

    if(readyToTests):
        modify_data, original_data = st.columns((1,1))
        with original_data:
            st.title("Original data")
            trainX, validX, trainY, validY, X, Y = splitData(original_dataframe, original_option_use_to_predict,
                                                             value_to_predict)
            cs2,cs2_std, mae2, mse2, rmse2, rmsle2, model2 = testModel(model, trainX, validX, trainY, validY, X, Y, metrics)
            if cs2:
                st.metric(label="Mean cross-validation score:", value=str(round(cs2 * 100, 2)) + "%", delta=None, delta_color="off")

                st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
                st.metric(label="Standard deviation:", value=str(round(cs2_std, 2)), delta=None, delta_color="off")
                st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

            if mae2:
                st.metric(label="Mean cross-validation MAE:", value=str(round(mae2, 2)), delta=None, delta_color="off")
                st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

            if mse2:
                st.metric(label="Mean cross-validation MSE", value=str(round(mse2, 2)), delta=None, delta_color="off")
                st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

            if rmse2:
                st.metric(label="Mean cross-validation RMSE", value=str(round(rmse2, 2)), delta=None, delta_color="off")
                st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

            if rmsle2:
                st.metric(label="Mean cross-validation RMSLE", value=str(round(rmsle2, 2)), delta=None, delta_color="off")
                st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

            joblib.dump(model2, "my_original_model.pkl")
            get_binary_file_downloader_html("my_original_model.pkl", "model2")

        with modify_data:
            st.title("Data with preprocessing")
            trainX, validX, trainY, validY, X, Y = splitData(my_dataframe,option_use_to_predict,value_to_predict)
            cs,cs_std, mae, mse, rmse, rmsle, model = testModel(model, trainX, validX, trainY, validY, X, Y , metrics)

            if cs:
                st.metric(label="Mean cross-validation score:", value=str(round(cs * 100, 2)) + "%", delta=str(round((cs-cs2) * 100, 3)) + "%")
                st.metric(label="Standard deviation:", value=str(round(cs_std, 2)), delta=round(cs_std - cs2_std, 3), delta_color="inverse")
            if mae:
                st.metric(label="Mean cross-validation MAE:", value=str(round(mae, 2)), delta=round(mae-mae2, 3), delta_color="inverse")
            if mse:
                st.metric(label="Mean cross-validation MSE", value=str(round(mse, 2)), delta=round(mse-mse2, 3), delta_color="inverse")
            if rmse:
                st.metric(label="Mean cross-validation RMSE", value=str(round(rmse, 2)), delta=round(rmse-rmse2, 3), delta_color="inverse")
            if rmsle:
                st.metric(label="Mean cross-validation RMSLE", value=str(round(rmsle, 2)), delta=round(rmsle-rmsle2, 3), delta_color="inverse")

            joblib.dump(model, "my_model.pkl")
            get_binary_file_downloader_html("my_model.pkl", "model")

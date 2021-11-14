import streamlit as st
import os
import base64
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

if __name__ != "__main__":
    PWD = os.getcwd()
    with open('widget.json', 'r') as openjson:
        json_widget_saver = json.load(openjson)
        pass
    openjson.close()


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def save_df_to_csv(df):
    df = df.to_csv(PWD+'/data.csv')


def save_uploadedfile(uploadedfile):
    # os.remove(pwd+'/output.pdf')
    with open(PWD + '/data.csv', "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} on Server".format(uploadedfile.name))

def download_csv(df):

    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    bin_file="csv.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{bin_file}">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)


def print_chart(my_dataframe, active_coefficient, coefficient_to_compere):

    array = np.array([my_dataframe[active_coefficient], my_dataframe[coefficient_to_compere]]).T
    if active_coefficient == coefficient_to_compere:
        coefficient_to_compere += '^'
    df = pd.DataFrame(
        array,
        columns=[active_coefficient, coefficient_to_compere])
    st.vega_lite_chart(df, {
        'mark': {'type': 'circle', 'tooltip': True},
        'encoding': {
            'x': {'field': active_coefficient, 'type': 'quantitative'},
            'y': {'field': coefficient_to_compere, 'type': 'quantitative'}
        },
    })


def resetWidgets():

    json_widget_saver['active_coefficient'] = ""
    json_widget_saver['change_value_btn'] = ""
    json_widget_saver['scale_btn'] = ""
    json_widget_saver['resize_range_btn'] = ""
    json_widget_saver['missing_value_btn'] = ""
    json_widget_saver['apply_scaler'] = ""
    json_widget_saver['upload_file'] = ""
    json_widget_saver['value_to_predict'] = ""
    json_widget_saver['base_dataset'] = ""

    with open("widget.json", "w") as outfile:
        json.dump(json_widget_saver, outfile)
    outfile.close()


def saveWidgets():

    with open("widget.json", "w") as outfile:
        json.dump(json_widget_saver, outfile)
    outfile.close()

def savePreprocesingButtons(preprocessing):
    if (preprocessing == 'Change Value'):
        json_widget_saver['change_value_btn'] = "1"
        json_widget_saver['scale_btn'] = ""
        json_widget_saver['missing_value_btn'] = ""
        json_widget_saver['resize_range_btn'] = ""
        saveWidgets()

    if (preprocessing == 'Resize Range'):
        json_widget_saver['change_value_btn'] = ""
        json_widget_saver['scale_btn'] = ""
        json_widget_saver['missing_value_btn'] = ""
        json_widget_saver['resize_range_btn'] = "1"
        saveWidgets()

    if (preprocessing == 'Scale'):
        json_widget_saver['change_value_btn'] = ""
        json_widget_saver['scale_btn'] = "1"
        json_widget_saver['missing_value_btn'] = ""
        json_widget_saver['resize_range_btn'] = ""
        saveWidgets()
    if (preprocessing == 'Missing Value Strategy'):
        json_widget_saver['change_value_btn'] = ""
        json_widget_saver['scale_btn'] = ""
        json_widget_saver['missing_value_btn'] = "1"
        json_widget_saver['resize_range_btn'] = ""
        saveWidgets()

def resizeColumn(my_dataframe,active_coefficient):

    r_min = min(my_dataframe[active_coefficient])
    r_max = max(my_dataframe[active_coefficient])

    t_min = st.text_input('Minimal Value')
    t_max = st.text_input('Maximum Value')
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

def refreshDataFrameWidget(dataFrameWidget,my_dataframe):
    dataFrameWidget.empty()
    dataFrameWidget.dataframe(my_dataframe)
    my_dataframe.to_csv(PWD + '/data.csv', index=False)
    saveWidgets()

def testModel(model,trainX, validX, trainY, validY):

    predictions = model.predict(validX)
    lin_mse = mean_squared_error(predictions,validY)
    st.title("MSE: "+str(lin_mse))
    lin_rmse = np.sqrt(lin_mse)
    st.title("RMSE: "+str(lin_rmse))
    st.title('Model Score: '+str(model.score(validX, validY)*100)+"%")
    #st.title(model.predict(np.array([[0.0381,0.0507]])))

def createModel(my_dataframe,option_use_to_predict,value_to_predict,algorithm_model):
    column_number = len(my_dataframe)
    #option_use_to_predict.append(value_to_predict)
    df = my_dataframe[option_use_to_predict]


    #trainX, validX, trainY, validY = train_test_split(df,df[active_coefficient], test_size=0.2, random_state=42)
    finaltrainX, finaltestX, finaltrainY, finaltestY = train_test_split(df[option_use_to_predict],my_dataframe[value_to_predict], test_size=0.2, random_state=42)
   # split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    #for train_index, test_index in split.split(df, df[active_coefficient]):
     #   strat_train_set = df.loc[train_index]
     #   strat_test_set = df.loc[test_index]

   # print(strat_train_set[active_coefficient].value_counts() / len(strat_train_set))
   # print(strat_test_set[active_coefficient].value_counts() / len(strat_test_set))

    #finaltrainX = strat_train_set.drop(active_coefficient, axis=1)
    #finaltrainY = strat_train_set[active_coefficient]

    #finaltestX = strat_test_set.drop(active_coefficient, axis=1)
    #finaltestY = strat_test_set[active_coefficient]


    if algorithm_model == 'LinearRegression':
        lin_reg = LinearRegression()
        lin_reg.fit(finaltrainX, finaltrainY)
        testModel(lin_reg, finaltrainX, finaltestX, finaltrainY, finaltestY)
    elif algorithm_model == 'RandomForestRegressor':
        forest_reg = RandomForestRegressor()
        forest_reg.fit(finaltrainX, finaltrainY)
        testModel(forest_reg, finaltrainX, finaltestX, finaltrainY, finaltestY)
    elif algorithm_model == "KNeighborsClassifier":
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(finaltrainX, finaltrainY)
        testModel(neigh, finaltrainX, finaltestX, finaltrainY, finaltestY)
    elif algorithm_model == "GaussianNB":
        gnb = GaussianNB()
        y_pred = gnb.fit(finaltrainX,finaltrainY)
        testModel(gnb, finaltrainX, finaltestX, finaltrainY, finaltestY)
    elif algorithm_model == "KMeans":
        kmeans = KMeans(n_clusters=2, random_state=0).fit(finaltrainX)

def normalized(dataframe,numerical_cols):
    norm = MinMaxScaler().fit(dataframe[numerical_cols])
    return pd.DataFrame( norm.transform(dataframe[numerical_cols]))

def standarization(dataframe,numerical_cols):
    train_stand = dataframe.copy()

    for i in numerical_cols:
        scale = StandardScaler().fit(train_stand[[i]])
        train_stand[i] = scale.transform(train_stand[[i]])
    return train_stand




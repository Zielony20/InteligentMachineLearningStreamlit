import streamlit as st
import os
import base64
import numpy as np
import pandas as pd
import json

if __name__!="__main__":
    print(__name__)
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
    json_widget_saver['missing_value_btn'] = ""
    with open("widget.json", "w") as outfile:
        json.dump(json_widget_saver, outfile)
    outfile.close()


def saveWidgets():

    with open("widget.json", "w") as outfile:
        json.dump(json_widget_saver, outfile)
    outfile.close()

def changeValueInColumn(my_dataframe,active_coefficient,ValueToChange,NewValue,DataType):

    my_dataframe[active_coefficient] = my_dataframe[active_coefficient].replace([ValueToChange], NewValue)
    #for i in my_dataframe[active_coefficient]:

    return my_dataframe


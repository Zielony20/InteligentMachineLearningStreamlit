import streamlit as st
import pandas as pd
import os
import base64
from Functions.JsonHandler import saveWidgets, json_widget_saver

if __name__ != "__main__":
    PWD = os.getcwd()

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

def save_df_to_csv(df):
    df.to_csv(PWD+'/data.csv', index=False)


def save_uploadedfile(uploadedfile):
    with open(PWD + '/data.csv', "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} on Server".format(uploadedfile.name))

def download_csv(df):
    csv = convert_df(df) #df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    bin_file="csv.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{bin_file}">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)

def saveAll(dataFrameWidget, my_dataframe ,rerun=False, active_coefficient=False):
    dataFrameWidget.empty()
    dataFrameWidget.dataframe(my_dataframe)
    my_dataframe.to_csv(PWD + '/data.csv', index=False)
    if(active_coefficient):
        json_widget_saver['active_coefficient'] = active_coefficient
    saveWidgets()
#    return dataFrameWidget
    if(rerun):
        st.experimental_rerun()

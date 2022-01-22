import streamlit as st
import pandas as pd
import os
import base64
from Functions.JsonHandler import saveWidgets, json_widget_saver
from Functions.PreprocessingFunctions import dropColumn
if __name__ != "__main__":
    PWD = os.getcwd()
    Slash = '/'



def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

def save_df_to_csv(df, name="data"):
    df.to_csv(PWD+'/'+name+'.csv', index=False)


def save_uploadedfile(uploadedfile,name="data"):
    with open(PWD + '/'+name+'.csv', "wb") as f:
        f.write(uploadedfile.getbuffer())
    #return st.success("Saved File:{} on Server".format(uploadedfile.name))

def download_csv(df):
    csv = convert_df(df) #df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    bin_file="csv.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{bin_file}">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)

def loadCsv(file):
    dataframe = pd.read_csv(PWD + '/' + file, index_col=None)
    if( "Unnamed: 0" in dataframe.columns):
        dataframe,_,_ = dropColumn(dataframe, "Unnamed: 0", dataframe.columns)
    return dataframe

def saveLastChange(my_dataframe,active_coefficient=False):
    my_dataframe.to_csv(PWD + '/lastchange.csv', index=False)
    if (active_coefficient):
        json_widget_saver['active_coefficient'] = active_coefficient
    saveWidgets()


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


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(PWD+Slash+bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    st.markdown(href, unsafe_allow_html=True)

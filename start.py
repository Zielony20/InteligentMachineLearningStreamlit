import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
from functions import *
from loadInterface import *

st.set_page_config(layout="wide")


upload_csv = st.file_uploader("put csv file", type='csv')

if (upload_csv is None):
    print("resetuje")
    resetWidgets()

if (json_widget_saver['upload_file'] == "1"):
    print(upload_csv)
    loadInterface()

elif (upload_csv is not None):
    print("wykonuje sie")
    save_uploadedfile(upload_csv)
    json_widget_saver['upload_file'] = "1"
    saveWidgets()

    loadInterface()


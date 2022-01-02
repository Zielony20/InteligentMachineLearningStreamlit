import streamlit as st
from Functions.FileSystemFunctions import save_uploadedfile, save_df_to_csv
from Functions.JsonHandler import *
from loadInterface import loadInterface
from sklearn import datasets
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
dataset_names = ['Own dataset', 'Iris plants dataset', 'Diabetes dataset', 'Wine recognition dataset', 'Breast cancer wisconsin (diagnostic dataset)']
sklearn_dataset = st.selectbox(
        'Which dataset use?',
        dataset_names)

if sklearn_dataset == 'Own dataset':
    example_dataset = False
    upload_csv = st.file_uploader("put csv file", type='csv')
elif sklearn_dataset in dataset_names:
    if sklearn_dataset != json_widget_saver['base_dataset'] and json_widget_saver['upload_file'] == "1":
        resetWidgets()
    example_dataset = True
    upload_csv = None
else:
    example_dataset = False
    upload_csv = None

if (upload_csv is None and example_dataset == False):
    resetWidgets()

if (json_widget_saver['upload_file'] == "1"):
    loadInterface()

elif (upload_csv is not None and example_dataset == False ):

    save_uploadedfile(upload_csv)
    json_widget_saver['upload_file'] = "1"
    saveWidgets()
    loadInterface()

elif example_dataset == True:

    if sklearn_dataset == 'Iris plants dataset':
        my_data = datasets.load_iris(return_X_y=False, as_frame=True)
    elif sklearn_dataset == 'Diabetes dataset':
        my_data = datasets.load_diabetes(return_X_y=False, as_frame=True)
    elif sklearn_dataset == 'Wine recognition dataset':
        my_data = datasets.load_wine(return_X_y=False, as_frame=True)
    elif sklearn_dataset == 'Breast cancer wisconsin (diagnostic dataset)':
        my_data = datasets.load_breast_cancer(return_X_y=False, as_frame=True)
    elif sklearn_dataset == 'Linnerrud dataset':
        my_data = datasets.load_linnerud(return_X_y=False, as_frame=True)


    data = np.c_[my_data.data, my_data.target]
    columns = np.append(my_data.feature_names, "target")
    save_df_to_csv(pd.DataFrame(data, columns=columns))
    json_widget_saver['base_dataset'] = sklearn_dataset
    json_widget_saver['upload_file'] = "1"
    saveWidgets()
    loadInterface()


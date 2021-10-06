import streamlit as st
import os
import base64
import numpy as np
import pandas as pd
PWD = os.getcwd()


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


def save_uploadedfile(uploadedfile):
    # os.remove(pwd+'/output.pdf')
    with open(PWD + '/data.csv', "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} on Server".format(uploadedfile.name))


def print_chart(my_dataframe, active_coefficient, coefficient_to_compere):
    array = np.array([my_dataframe[active_coefficient], my_dataframe[coefficient_to_compere]]).T
    print(array.dtype)
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

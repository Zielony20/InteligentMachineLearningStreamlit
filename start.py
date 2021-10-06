import streamlit as st
from PIL import Image
import pandas as pd
import base64
import matplotlib.pyplot as plt
from translate import Translator
from functions import *
from loadInterface import *



st.set_page_config(layout="wide")

upload_csv = st.file_uploader("put csv file")
if (upload_csv is None):
    pass
if (upload_csv is not None):
    save_uploadedfile(upload_csv)
    loadInterface()

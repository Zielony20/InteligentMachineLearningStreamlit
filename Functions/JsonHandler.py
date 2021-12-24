import os
import json
import streamlit as st

if __name__ != "__main__":
    PWD = os.getcwd()
    with open(PWD+'/widget.json', 'r') as openjson:
        json_widget_saver = json.load(openjson)
        pass
    openjson.close()

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

    with open("../widget.json", "w") as outfile:
        json.dump(json_widget_saver, outfile)
    outfile.close()

def saveWidgets():

    with open("../widget.json", "w") as outfile:
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

def refreshDataFrameWidget(dataFrameWidget,my_dataframe):
    dataFrameWidget.empty()
    dataFrameWidget.dataframe(my_dataframe)
    my_dataframe.to_csv(PWD + '/data.csv', index=False)
    saveWidgets()

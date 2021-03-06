from Functions import PreprocessingFunctions as pf
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as sc
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def print_chart_with_target(my_dataframe, active_coefficient, coefficient_to_compere, target):
    df = my_dataframe.loc[:,[active_coefficient, coefficient_to_compere, target]]

    st.vega_lite_chart(df, {
        'mark': {'type': 'circle', 'tooltip': False},
        'encoding': {
            'x': {'field': active_coefficient, 'type': 'quantitative'},
            'y': {'field': coefficient_to_compere, 'type': 'quantitative'},
            'size': {'field': target, 'type': 'nominal'},
            'color': {'field': target, 'type': 'nominal'}
        },
    })

def print_chart(my_dataframe, active_coefficient, coefficient_to_compere):
    class_object_cols = list(pf.getClassificationColums(my_dataframe))
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

def print_chart2(my_dataframe, active_coefficient, coefficient_to_compere, target, color="Bluered"):

    fig = px.scatter(my_dataframe, x=active_coefficient, y=coefficient_to_compere, log_x=False, color=target, color_continuous_scale=color) #Bluered #Rainbow
    st.plotly_chart(fig, use_container_width=True)

def print_chart_with_options(my_dataframe, active_coefficient,targets,numeric_object_cols):
    c1, c2 = st.columns((5, 1))
    with c2:
        to_compere = st.selectbox("To compare", numeric_object_cols)
        target = st.selectbox("Target  ", numeric_object_cols)
        color = st.selectbox("Color Pattern: ", ["Bluered", "Rainbow"])
    with c1:
        print_chart2(my_dataframe, active_coefficient, to_compere, target, color)

def histSimilarity(function):

    list = {}
    list_p = {}
    s, p = sc.kstest(function, "norm")
    list["Normal"] = s
    list_p["Normal"] = p
    s, p = sc.kstest(function, "uniform")
    list["Uniform"] = s
    list_p["Uniform"] = p
    s, p = sc.kstest(function, "cauchy")
    list["Cauchy"] = s
    list_p["Cauchy"] = p
    s, p = sc.kstest(function, "expon")
    list["Expon"] = s
    list_p["Expon"] = p
    s, p = sc.kstest(function, "laplace")
    list["Laplace"] = s
    list_p["Laplace"] = p
    s, p = sc.kstest(function, "rayleigh")
    list["Rayleigh"] = s
    list_p["Rayleigh"] = p


    dic = dict(sorted(list_p.items(),key=lambda x:x[1],reverse=True))

    #p_values = dict(sorted(list_p.items(), key=lambda x: x[1]))
    st.text("Most similar distributions:")

    counter = 0
    for dist,stat in dic.items():
        if(counter == 5):
            break
        if(stat > 0):
            counter += 1
            st.text("{}.{} p value = {:0.2e}".format(str(counter),dist,list_p[dist])  )
        if(counter == 0 and stat == 0):
            st.text("distributions not recognized")
            break



def histogramWithKomogorov(active_coefficient,my_dataframe):
    kolmogorov, histPlace, stats = st.columns((2, 4, 1))
    with histPlace:
        int_val = [.01]  # st.number_input('hist bins', value=1, step=1,format="%.2f")
        group_labels = [active_coefficient]
        hist_data = [my_dataframe[active_coefficient].to_numpy()]

        fig = px.histogram(my_dataframe[active_coefficient], x=active_coefficient, facet_col_spacing=1,
                           marginal="violin", histnorm=None, barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

        with kolmogorov:
            st.text('\n')
            st.text('\n')
            st.text('\n')
            histSimilarity(my_dataframe[active_coefficient].to_numpy())

        with stats:
            st.text('\n')
            st.text('\n')
            st.text('\n')
            # st.text('count:' + str(round(my_dataframe.describe().count()[active_coefficient], 2)))
            st.text('mean:' + str(round(my_dataframe[active_coefficient].mean(), 2)))
            st.text('std:' + str(round(my_dataframe[active_coefficient].std(), 2)))
            st.text('max:' + str(round(my_dataframe[active_coefficient].max(), 2)))
            st.text('min:' + str(round(my_dataframe[active_coefficient].min(), 2)))
            # st.text('unique:' + str(round(my_dataframe.describe(include='all').unique()[active_coefficient], 2)))
            # st.text('freq:' + str(round(my_dataframe.describe(include='all').freq(), 2)))
            print(my_dataframe[active_coefficient].describe())


def simpleCharts(my_dataframe, active_coefficient,targets,numeric_object_cols):
    c1, c2, c3, c4 = st.columns((1, 1, 1, 1))
    counter = 1
    for coefficient in numeric_object_cols:
        with c1:
            if counter == 1:
                print_chart(my_dataframe, active_coefficient, coefficient)
        with c2:
            if counter == 2:
                print_chart(my_dataframe, active_coefficient, coefficient)
        with c3:
            if counter == 3:
                print_chart(my_dataframe, active_coefficient, coefficient)
        with c4:
            if counter == 4:
                print_chart(my_dataframe, active_coefficient, coefficient)

        counter += 1
        if counter >= 5:
            counter = 1

def colorsCharts(my_dataframe, active_coefficient, targets, numeric_object_cols):
    target = st.selectbox("Target", numeric_object_cols)#targets
    color = st.selectbox("Color pattern", ["Bluered", "Rainbow"])
    cc1, cc2, cc3, cc4 = st.columns((1, 1, 1, 1))

    counter = 1

    for coefficient in numeric_object_cols:

        with cc1:
            if counter == 1:
                print_chart2(my_dataframe, active_coefficient, coefficient, target, color)
        with cc2:
            if counter == 2:
                print_chart2(my_dataframe, active_coefficient, coefficient, target, color)
        with cc3:
            if counter == 3:
                print_chart2(my_dataframe, active_coefficient, coefficient, target, color)
        with cc4:
            if counter == 4:
                print_chart2(my_dataframe, active_coefficient, coefficient, target, color)

        counter += 1
        if counter >= 5:
            counter = 1

def crossCharts(my_dataframe,targets):
    target = st.selectbox("Target ", targets)
    df = my_dataframe
    numeric_object_cols = pf.getNumericalColumns(my_dataframe)
    fig = px.scatter_matrix(df, dimensions=numeric_object_cols,
                            color=target, color_continuous_scale="Bluered") #agsunset #Bluered #Rainbow
    fig.update_layout(
        title="",
        dragmode='select',
        width=600,
        height=600,
        hovermode='closest',

    )
    st.plotly_chart(fig, use_container_width=True)

def charts(my_dataframe,active_coefficient):

    targets = pf.getClassificationColums(my_dataframe)
    numeric_object_cols = pf.getNumericalColumns(my_dataframe)

    border = 7

    if(len(targets)==0):
        if(len(numeric_object_cols)<=border):
            simpleCharts(my_dataframe, active_coefficient, targets, numeric_object_cols)
        elif (len(numeric_object_cols) > border):
            simpleCharts(my_dataframe, active_coefficient, targets, numeric_object_cols)
    elif(len(targets)>0):
        if (len(numeric_object_cols) <= border):
            colorsCharts(my_dataframe, active_coefficient, targets, numeric_object_cols)
            crossCharts(my_dataframe, targets)
        elif (len(numeric_object_cols) > border):
            #simpleCharts(my_dataframe, active_coefficient, targets, numeric_object_cols)
            colorsCharts(my_dataframe, active_coefficient, targets, numeric_object_cols)
            print_chart_with_options(my_dataframe, active_coefficient, targets, numeric_object_cols)

def pieChart(my_dataframe,active_coefficient,values):
    df = my_dataframe
    fig = px.pie(df, values=values, names=active_coefficient)
    st.plotly_chart(fig, use_container_width=True)

def counterPieChart(my_dataframe,active_coefficient):
    df = my_dataframe
    labels = df[active_coefficient].unique()
    values = df[active_coefficient].value_counts().values
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    st.plotly_chart(fig, use_container_width=True)


def correlationHeatmap(my_dataframe):
    fig = px.imshow(my_dataframe.corr(), color_continuous_scale="Bluered")
    #ax = sns.heatmap(my_dataframe.corr(), annot=True)
    st.plotly_chart(fig, use_container_width=True)




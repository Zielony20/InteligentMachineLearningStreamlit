from Functions import PreprocessingFunctions as pf
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as sc
import plotly.express as px
import plotly.graph_objects as go

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

def print_chart_with_target(my_dataframe, active_coefficient, coefficient_to_compere, target):
    #print(my_dataframe.loc[:,[active_coefficient, coefficient_to_compere, target]])
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

def print_chart2(my_dataframe, active_coefficient, coefficient_to_compere, target):

    fig = px.scatter(my_dataframe, x=active_coefficient, y=coefficient_to_compere, log_x=False, color=target)
    st.plotly_chart(fig, use_container_width=True)


def histSimilarity(function):

    list = {}
    s ,_ = sc.kstest(function, "norm")
    list["Normal"] = s
    s, _ = sc.kstest(function, "uniform")
    list["Uniform"] = s
    s, _ = sc.kstest(function, "cauchy")
    list["Cauchy"] = s
    s, _ = sc.kstest(function, "expon")
    list["Expon"] = s
    s, _ = sc.kstest(function, "laplace")
    list["Laplace"] = s
    s, _ = sc.kstest(function, "rayleigh")
    list["Rayleigh"] = s

    dic = dict(sorted(list.items(),key= lambda x:x[1]))

    st.text("Most similar distributions:")

    counter = 0
    for dist,stat in dic.items():
        if(counter == 5):
            break
        if(stat < 1):
            counter += 1
            st.text(str(counter)+"."+dist)
        if(counter == 0 and stat == 1):
            st.text("distributions not recognized")

def histogramWithKomogorov(active_coefficient,my_dataframe):
    kolmogorov, histPlace, _ = st.columns((1, 4, 1))
    with histPlace:
        int_val = [.01]  # st.number_input('hist bins', value=1, step=1,format="%.2f")
        group_labels = [active_coefficient]
        hist_data = [my_dataframe[active_coefficient].to_numpy()]
        # Create distplot with custom bin_size
        #        fig = ff.create_distplot(
        #           hist_data, group_labels, bin_size=int_val, histnorm="probability density")
        # Plot!
        #      st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(my_dataframe[active_coefficient], x=active_coefficient, facet_col_spacing=1,
                           marginal="violin", histnorm=None, barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

        with kolmogorov:
            st.text('\n')
            st.text('\n')
            st.text('\n')
            histSimilarity(my_dataframe[active_coefficient].to_numpy())


def chartsCoordinator(active_coefficient,my_dataframe,numeric_object_cols):
    targets = pf.getClassificationColums(my_dataframe)

    if (len(numeric_object_cols) > 8 and len(targets) > 0):
        comparisonCharts(active_coefficient, my_dataframe, numeric_object_cols, [])
        c1, c2 = st.columns((5, 1))
        with c2:
            to_compere = st.selectbox("To compare", numeric_object_cols)
            target = st.selectbox("Target", targets)
        with c1:
            print_chart2(my_dataframe, active_coefficient, to_compere, target)
    else:
        target = st.selectbox("Target", targets)
        comparisonCharts(active_coefficient, my_dataframe, numeric_object_cols, targets)

def comparisonCharts(active_coefficient,my_dataframe,numeric_object_cols,targets):
    c1, c2, c3, c4 = st.columns((1, 1, 1, 1))
    columns_array = [c1, c2, c3, c4]
    counter = 1

    for coefficient in numeric_object_cols:
        if counter == 1:
            with c1:
                if (len(targets) == 0):
                    print_chart(my_dataframe, active_coefficient, coefficient)
                else:
                    print_chart2(my_dataframe, active_coefficient, coefficient, target)
        if counter == 2:
            with c2:
                if (len(targets) == 0):
                    print_chart(my_dataframe, active_coefficient, coefficient)
                else:
                    print_chart2(my_dataframe, active_coefficient, coefficient, target)
        if counter == 3:
            with c3:
                if (len(targets) == 0):
                    print_chart(my_dataframe, active_coefficient, coefficient)
                else:
                    print_chart2(my_dataframe, active_coefficient, coefficient, target)
        if counter == 4:
            with c4:
                if (len(targets) == 0):
                    print_chart(my_dataframe, active_coefficient, coefficient)
                else:
                    print_chart2(my_dataframe, active_coefficient, coefficient, target)

        counter += 1
        if counter >= 5:
            counter = 1


def crossCharts(my_dataframe,target):

    df = my_dataframe
    numeric_object_cols = pf.getNumericalColumns(my_dataframe)
    fig = px.scatter_matrix(df, dimensions=numeric_object_cols,
                            color=target)
    fig.update_layout(
        title="",
        dragmode='select',
        width=600,
        height=600,
        hovermode='closest',
    )
    st.plotly_chart(fig, use_container_width=True)

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


# Zaimplementować to:
# https://plotly.com/python/plotly-express/



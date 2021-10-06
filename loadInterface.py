import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split


def loadInterface():
    my_dataframe = pd.read_csv(PWD + '/data.csv')
    st.dataframe(my_dataframe)

    print("Variables types")
    print(my_dataframe.dtypes)

    s = (my_dataframe.dtypes == 'object')
    object_cols = list(s[s].index)

    print("Categorical variables:")
    print(object_cols)

    n = (my_dataframe.dtypes == 'int64')
    numeric_object_cols = list(n[n].index)

    print("Numerical Variable")
    print(numeric_object_cols)

    f = (my_dataframe.dtypes == 'float64')
    float_object_cols = list(f[f].index)

    print("Float Variable")
    print(float_object_cols)

    # cols_with_missing = [col for col in my_dataframe.columns if my_dataframe[col].isnull().any()]
    # my_dataframe.drop(cols_with_missing, axis=1, inplace=True)
    column_names = my_dataframe.columns
    print(column_names)
    variable_numbers = len(my_dataframe.columns)

    active_coefficient = ''

    with st.sidebar:
        st.info(
            "🎈 **NEW:** Add your own code template to this site! [Guide]("
            "https://github.com/jrieke/traingenerator#adding-new-templates) "
        )
        buttons = list()
        buttons = st.columns(variable_numbers)
        index = 0
        code = -1

        for i in buttons:
            index += 1

            # st.title(column_names[index-1])
            if st.button(column_names[index - 1]):
                active_coefficient = column_names[index - 1]

    column_number = len(my_dataframe)

    # train_set = my_dataframe[0:column_number*0.8]
    # test_set = my_dataframe[column_number*0.8+1:column_number*0.8]

    X_train, X_valid, y_train, y_valid = train_test_split(my_dataframe[0:column_number], range(column_number),
                                                          train_size=0.8, test_size=0.2, random_state=42)



    if active_coefficient != '':

        # Space out the maps so the first one is 2x the size of the other three
        c1, c2, c3, c4 = st.columns((1, 1, 1, 1))
        columns_array = [c1, c2, c3, c4]
        counter = 0
        for coefficient in numeric_object_cols:

            if counter==1:
                with c1:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 2:
                with c2:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 3:
                with c3:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 4:
                with c4:
                    print_chart(my_dataframe, active_coefficient, coefficient)

            counter += 1
            if counter >= 5:
                counter = 1


        for coefficient in float_object_cols:
            if counter == 1:
                with c1:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 2:
                with c2:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 3:
                with c3:
                    print_chart(my_dataframe, active_coefficient, coefficient)
            if counter == 4:
                with c4:
                    print_chart(my_dataframe, active_coefficient, coefficient)

            counter += 1
            if counter >= 5:
                counter = 1



    # Use the full page instead of a narrow central column




'''
df = pd.DataFrame(
    np.random.randn(200, 2),
    columns=['a', 'b'])

st.vega_lite_chart(df, {
    'mark': {'type': 'circle', 'tooltip': True},
    'encoding': {
        'x': {'field': 'a', 'type': 'quantitative'},
        'y': {'field': 'b', 'type': 'quantitative'}
        
    },
})
'''

'''

df = pd.DataFrame(
   np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
   columns=['lat', 'lon'])
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=37.76,
        longitude=-122.4,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=df,
           get_position='[lon, lat]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ],
))

'''

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def predictCoefficient(model,columns):

    cols = []
    for c in columns:
        cols.append(st.text_input(c))
    if st.button("predict") and len(cols)==len(columns):
        st.text(model.predict(cols))

def testModel(model,trainX, validX, trainY, validY,X,Y, metrics):

    cvs = cross_val_score(model, X, Y, cv=5)
    print(round(cvs.mean(), 2))
    st.title('Mean cross-validation score: '+str(round(cvs.mean()*100, 2))+"%")
    print(cvs)
    st.title('Standard deviation of scores: ' + str(round(cvs.std(), 2)))
    #st.title(model.predict(np.array([[0.0381,0.0507]])))

    model = model.fit(trainX, trainY)

    predictions = model.predict(validX)
    #print(predictions)
    #print(validX)
    model_mse = mean_squared_error(predictions, validY)
    model_rmse = np.sqrt(model_mse)
    model_mae = mean_absolute_error(predictions, validY)
    model_rmsle = np.log(np.sqrt(model_mse))
    model_r2 = r2_score(predictions, validY)
    #st.title('Model Score: ' + str(round(model.score(validX, validY) * 100, 2)) + "%")
    if ("MSE" in metrics):
        st.title("MSE: " + str(round(model_mse, 2)))
    if ("RMSE" in metrics):
        st.title("RMSE: " + str(round(model_rmse, 2)))
    if ("MAE" in metrics):
        st.title("MAE: " + str(round(model_mae, 2)))
    if ("RMSLE" in metrics):
        st.title("RMSLE: " + str(round(model_rmsle, 2)))
    if ("R2 Squared" in metrics):
        st.title("R2 Squared: " + str(round(model_r2, 2)))

    import joblib
    from Functions.FileSystemFunctions import get_binary_file_downloader_html
    model = model.fit(X,Y)
    joblib.dump(model, "my_model.pkl")
    get_binary_file_downloader_html("my_model.pkl", "model")


    #predictCoefficient(model, X.columns)

def createModel(my_dataframe,option_use_to_predict,value_to_predict,algorithm_model,metrics):
    column_number = len(my_dataframe)

    df = my_dataframe[option_use_to_predict]
    X = df[option_use_to_predict]
    Y = my_dataframe[value_to_predict]
    finaltrainX, finaltestX, finaltrainY, finaltestY = train_test_split(df[option_use_to_predict],my_dataframe[value_to_predict], test_size=0.3, random_state=42)


    if algorithm_model == 'LinearRegression':
        first, second, third, forth, fifth = st.columns((1, 1, 1, 1, 1))
        with first:
            fit_intercept = st.selectbox("fit_intercept", [True, False])
        if st.button("Create Model"):
            lin_reg = LinearRegression(fit_intercept=fit_intercept, normalize=False, copy_X=True, n_jobs=None, positive=False)
            #lin_reg.fit(finaltrainX, finaltrainY)
            testModel(lin_reg, finaltrainX, finaltestX, finaltrainY, finaltestY,X,Y,metrics)

    elif algorithm_model == 'RandomForestRegressor':
        first, second, third, forth, fifth = st.columns((1, 1, 1, 1, 1))
        with first:
            criterion = st.selectbox("criterion", ["squared_error", "absolute_error", "poisson"])
        with second:
            min_samples_leaf = st.selectbox("min_samples_leaf", [1, 2, 3, 4, 5])
        with third:
            max_depth = st.selectbox("max_depth", [None, 1, 2, 3, 4, 5])
        if st.button("Create Model"):
            forest_reg = RandomForestRegressor(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
            #forest_reg.fit(finaltrainX, finaltrainY)
            testModel(forest_reg, finaltrainX, finaltestX, finaltrainY, finaltestY,X,Y, metrics)

    elif algorithm_model == 'DecisionTree':
        first, second, third, forth, fifth = st.columns((1, 1, 1, 1, 1))
        with first:
            criterion = st.selectbox("criterion",["gini","entropy"])
        with second:
            min_samples_leaf = st.selectbox("min_samples_leaf",[1,2,3,4,5])
        with third:
            max_depth = st.selectbox("max_depth", [None,1,2,3,4,5])
        if st.button("Create Model"):
            decisionTree = DecisionTreeClassifier(criterion=criterion,splitter="best",max_depth=max_depth,min_samples_split=2,
                                                  min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=0.0,max_features=None,
                                                  random_state=42, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                  class_weight=None,ccp_alpha=0.0)
            #decisionTree.fit(finaltrainX, finaltrainY)
            testModel(decisionTree, finaltrainX, finaltestX, finaltrainY, finaltestY,X,Y,metrics)

    elif algorithm_model == "KNeighborsClassifier":
        first, second, third, forth, fifth = st.columns((1, 1, 1, 1, 1))
        with first:
            algorithm = st.selectbox("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        with second:
            n_neighbors = st.selectbox("n_neighbors", [2, 3, 4, 5, 6, 7, 8, 9])
        with third:
            weights = st.selectbox("weights", ["uniform", "distance"])
        if st.button("Create Model"):
            neigh = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
            #neigh.fit(finaltrainX, finaltrainY)
            testModel(neigh, finaltrainX, finaltestX, finaltrainY, finaltestY,X,Y, metrics)


    elif algorithm_model == "SGDClassifier":
        sgd = SGDClassifier()#.fit(finaltrainX, finaltrainY.astype('int'))
        testModel(sgd, finaltrainX, finaltestX, finaltrainY.astype('int'), finaltestY.astype('int'),X,Y, metrics)

    elif algorithm_model == "LogisticRegression":
        lr = LogisticRegression(random_state=42)#.fit(finaltrainX, finaltrainY.astype('int'))
        testModel(lr, finaltrainX, finaltestX, finaltrainY.astype('int'), finaltestY.astype('int'),X,Y, metrics)

    elif algorithm_model == "GaussianNB":
        gnb = GaussianNB()
        y_pred = gnb.fit(finaltrainX,finaltrainY)
        testModel(gnb, finaltrainX, finaltestX, finaltrainY, finaltestY,X,Y, metrics)
    elif algorithm_model == "KMeans":
        kmeans = KMeans(n_clusters=2, random_state=0)#.fit(finaltrainX)
        testModel(kmeans, finaltrainX, finaltestX, finaltrainY, finaltestY,X,Y, metrics)



import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

st.write('''
# Random Forest Classifier
## Made by Muneebullah.
This app uses a Random Forest Classifier to predict the species of a given iris flower based on its se
''')
st.sidebar.header('Change Iris Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length, ', 4.3, 7.9, 5.0)
    sepal_width = st.sidebar.slider('Sepal_width, ', 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider('Petal_length, ', 0, 10, 3)
    petal_width = st.sidebar.slider('Petal width, ', 0.1, 1.8, 1.0)
    data = {'sepal_length': sepal_length, 'sepal_width': sepal_width, 'petal_length': petal_length, 'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('IRIS parameters')
st.write(df)

iris = sns.load_dataset('iris')

st.subheader('Iris dataset')
st.write(iris.head(10))

st.subheader('Plotly plots')
fig = px.scatter_3d(iris, x = 'sepal_length', y = 'petal_length', z = 'petal_width', color = 'species')
st.plotly_chart(fig)
fig = px.box(iris, x = 'species', y = 'petal_length', color = 'species')
st.plotly_chart(fig)


X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris['species']

model = RandomForestClassifier()
model.fit(X, y)

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
st.subheader('class labels and thier coressponding index number')
st.write(iris['species'].unique())

st.subheader('Prediction')
p = st.write(prediction[0])
st.write(p)

st.subheader('Prediction Probability')
st.write(prediction_proba)
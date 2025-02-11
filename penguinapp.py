import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write(""" 
# Penguin Prediction App
This app predicts the **Palmer panguin** species!

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""

[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)

""")

uploaded_file = st.sidebar.file_uploader(
    "Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox(
            'Island', ('Biscoe', 'Dream', 'Torgensen'))
        sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
        bill_lenght_mm = st.sidebar.slider(
            'Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider(
            'Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider(
            'Body mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'sex': sex,
                'bill_length_mm': bill_lenght_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g}

        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

url = "https://raw.githubusercontent.com/dataprofessor/code/refs/heads/master/streamlit/part3/penguins_cleaned.csv"

penguins_raw = pd.read_csv(url)
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

encode = ['sex', 'island']

for col in encode:
    df[col] = df[col].str.lower()
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df.rename(columns={
    'island_biscoe': 'island_Biscoe',
    'island_dream': 'island_Dream',
    'island_torgersen': 'island_Torgersen',
    'sex_male': 'sex_male',
    'sex_female': 'sex_female'
})

df = df[:1]


st.subheader('user_input_features')

if uploaded_file is not None:
    df = df.drop(columns=['species'])
    st.write(df)

else:
    st.write(
        'Awaiting CSV file to be uploaded. Currently using example input parameters (Shown below)')
    st.write(df)
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))
print("Columnas esperadas por el modelo:", load_clf.feature_names_in_)


prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)
st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction probability')
prediction_proba_df = pd.DataFrame(prediction_proba, columns=penguins_species)
st.write(prediction_proba_df)

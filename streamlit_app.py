import joblib
import streamlit as st
import numpy as np

model = joblib.load('final_model')
scaler = joblib.load('scaler')

def prix(model, carL, carW, curbW, cylinderN, engineSz, horseP, cityMPG, highwayMPG):
    """Prédit le prix d'une voiture"""
    x = np.array([carL, carW, curbW, cylinderN, engineSz, horseP, cityMPG, highwayMPG]).reshape(1,8)
    x_scaled = scaler.transform(x)
    return model.predict(x_scaled)[0]

st.set_page_config(layout='wide', page_title='CarPriceApp', page_icon=':car')

st.title("App prédiction prix d'une voiture")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Caractéristiques du vehicule :')
    carLength = st.slider('Longueur de la voiture (pouces)',140,220,175)
    carWidth = st.slider('Largeur de la voiture (pouces)',60,75,65)
    curbWeight = st.slider('Poids à vide (livres)',1480,4100,2555)

with col2:
    st.subheader('Caractéristiques du moteur :')
    cylinderNumber = st.selectbox('Nombre de cylindres', ('2','3','4','5','6','8','12'))
    engineSize = st.slider('Taille du moteur (pouces)',50,350,126)
    horsePower = st.slider('Puissance (ch.)',45,300,104)

with col3:
    st.subheader('Consommation du vehicule :')
    citympg = st.slider('Consommation en ville (Miles per Galon)',10,60,25)
    highwaympg = st.slider('Consommation autoroute (Miles per Galon)',15,60,30)

prediction = prix(model, carLength, carWidth, curbWeight, cylinderNumber, engineSize, horsePower, citympg, highwaympg)
st.write("Le prix estimé du vehicule est ",round(prediction), "$")

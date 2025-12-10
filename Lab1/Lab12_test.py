import streamlit as st

st.header("Hello World")
st.image("./images/iris.png")

st.sidebar.header("Menu")

choix = st.sidebar.selectbox("Select un model",['choix1','choix2','choix3'])

st.write("Vous avez choisi:",choix)

slider = st.sidebar.slider("Slider",0,10,8)

st.write("Valeur du slider:",slider)
import pickle
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from PIL import Image

#Funcion para procesar el dibujo
def guardarImagen(data):
    im = Image.fromarray(data.image_data)
    im = im.resize((28,28))
    #TODO borrar este image save, solo para el word
    #im.save("image.png")
    dataArray = np.asarray(im)
    dataArray = dataArray[:,:,1] /255.0
    dataArray = dataArray.reshape(1,28,28,1)  
    number = model.predict(dataArray)
    #Número a imprimir por pantalla
    st.text("Prediccion del modelo: ")
    argmax = np.argmax(number)
    st.text(argmax)

#Deserializar el modelo con pickle
with open("cnnModel.pkl","rb") as f:
    model = pickle.load(f)

# Título de la página
st.title('Prediccion de número')
st.markdown("Dibuja un número")


# Create a canvas component
#Este componente devuelve un array (280,280,4), son 3 canales RGB y uno de opacidad
data = st_canvas(
    background_color="#eee",height=420,width=420
)

#Boton para predecir
if st.button("Predict number"):
    #predict(data)
    guardarImagen(data)

    



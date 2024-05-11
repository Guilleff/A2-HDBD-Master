import pickle
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

#Funcion para procesar el dibujo
def predict(data):
    #Guadar el array como imagen
    im = Image.fromarray(data.image_data)
    #Escalar la imagen
    im = im.resize((28,28))
    #TODO borrar este image save, solo para el word
    #im.save("image.png")
    #Obtener el array escalado
    dataArray = np.asarray(im)
    #Seleccion de 1 capa y transformacion a escala de grises
    dataArray = dataArray[:,:,1] /255.0
    dataArray = dataArray.reshape(1,28,28,1)  
    #Prediccion del numero
    number = model.predict(dataArray)
    #Número a imprimir por pantalla
    st.text("Prediccion del modelo: ")
    argmax = np.argmax(number)
    st.text(argmax)

#Deserializar el modelo con pickle
with open("cnnModel.pkl","rb") as f:
    model = pickle.load(f)

#Informacion de la pagina
st.title('Actividad 2 Guillermo Fernandez')
st.subheader("Opcion C) Prediccion de un digito")
st.text("Se ha entrenado y desplegado un modelo capaz de reconocer dígitos manuscritos.\nSe ha utilizado el dataset MNIST incorporado en Keras")

#Este componente devuelve un array (420,420,4), son 3 canales RGB y uno de opacidad
data = st_canvas(
    background_color="#eee",height=420,width=420
)

#Boton para predecir
if st.button("Predict number"):
    predict(data)

    



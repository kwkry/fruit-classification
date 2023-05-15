import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

def main():
    # set up the Streamlit app
    st.write("Name: Vince Kurt C. Agripa")
    st.write("Section: CPE32S6")
    st.title("Fruit Classifier App")
    st.write("This is a fruit classification application, where it classes whether the fruit is an apple, banana, carambola, guava, kiwi, mango, muskmelon, orange, peach, pear, persimmon, pitaya, plum, pomegranate, tomatoes.")
    st.write("### Classify your fruit image!")
   
    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model('model_fruit_classificaiton.hdf5')
        return model
    
    def import_and_predict(image_data, model):
        size=(128,128)
        image = ImageOps.fit(image_data,size, Image.LANCZOS)
        image = np.asarray(image)
        image = image / 255.0
        img_reshape = np.reshape(image, (1, 150, 150, 3))
        prediction = model.predict(img_reshape)
        return prediction

    model = load_model()
    class_names = ["Apple", "Banana", "Carambola", 
                   "Guava", "Kiwi", "Mango", 
                   "Muskmelon", "Orange", "Peach", 
                   "Pear", "Persimmon", "Pitaya", 
                   "Plum", "Pomegranate", "Tomatoes"]
    

    file = st.file_uploader("Upload a fruit picture", type=["jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        string = "Prediction: " + class_name
        st.success(string)
 
if __name__ == "__main__":
    main()

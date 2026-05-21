import streamlit as st

import tensorflow as tf

from tensorflow.keras.preprocessing import image

from PIL import Image

import numpy as np

# Load CNN Model
model = tf.keras.models.load_model("cnn_model.h5")

# Title
st.title("Cat vs Dog Classification")

st.write("Upload a Cat or Dog Image")

# Upload File
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    # Open Image
    img = Image.open(uploaded_file)

    # Display Image
    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize Image
    img = img.resize((128,128))

    # Convert Image to Array
    img_array = image.img_to_array(img)

    # Expand Dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize Image
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)

    # Result
    if prediction[0][0] > 0.5:
        st.success("Prediction : Dog")
    else:
        st.success("Prediction : Cat")
import streamlit as st
from fastai.vision.all import *

st.title("Single Digit Prediction")
st.text("Built by Joel Suwanto")

def number_label(file_path):
    file_parts = str(file_path).split("/")

    return file_parts[-2]

single_digit_model = load_learner("single_digit_model (1).pkl")

uploaded_file = st.file_uploader("Upload an image of a digit...", type=["jpg", "png", "jpeg"])

def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = single_digit_model.predict(img)
    return pred_class


if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    prediction = predict(uploaded_file)
    st.subheader(f"Predicted Digit: {prediction}")


st.text("Built with Streamlit and FastAI")
    
    

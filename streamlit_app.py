import streamlit as st
from fastai.vision.all import *

st.title("Single Digit Prediction")
st.text("Built by Joel Suwanto")

def extract_number(path):
    path_parts_list = str(path).split("/")
    return path_parts_list[8]

single_digit_model = load_learner("mnist_model_fastai_287.pkl")

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



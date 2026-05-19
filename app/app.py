import streamlit as st

from image_model import ImageDetector
from text_model import TextDetector
from fusion_engine import multimodal_profile_analysis


st.set_page_config(
    page_title="Synthetic Profile Detector",
    layout="wide"
)

st.title(
    "Adversarial Multimodal AI Framework for Synthetic Profile Detection"
)

image_model = ImageDetector(
    "../models/fake_profile_detector.pth"
)

text_model = TextDetector(
    "../models/distilbert_text_model.pth"
)

uploaded_image = st.file_uploader(
    "Upload Profile Image",
    type=["jpg","jpeg","png"]
)

profile_bio = st.text_area(
    "Enter Profile Bio"
)

if st.button("Analyze Profile"):

    if uploaded_image and profile_bio:

        with open(
            "temp_image.jpg",
            "wb"
        ) as f:

            f.write(
                uploaded_image.getbuffer()
            )

        image_result = image_model.predict(
            "temp_image.jpg"
        )

        text_result = text_model.predict(
            profile_bio
        )

        fusion_result = multimodal_profile_analysis(
            image_result["prediction"],
            image_result["confidence"],
            text_result["prediction"],
            text_result["confidence"]
        )

        st.success(
            f"Image Prediction: {image_result['prediction']}"
        )

        st.info(
            f"Image Confidence: {image_result['confidence']}%"
        )

        st.success(
            f"Text Prediction: {text_result['prediction']}"
        )

        st.info(
            f"Text Confidence: {text_result['confidence']}%"
        )

        st.subheader(
            "Final Decision"
        )

        st.metric(
            "Verdict",
            fusion_result["verdict"]
        )

        st.metric(
            "Risk Score",
            fusion_result["risk_score"]
        )

    else:

        st.warning(
            "Please provide image and profile bio."
        )

import streamlit as st
import streamlit.components.v1 as components

from image_model import ImageDetector
from text_model import TextDetector
from fusion_engine import multimodal_profile_analysis


st.set_page_config(
    page_title="Cyber AI Dashboard",
    layout="wide"
)

with open(
    "styles.css"
) as css:

    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )


components.html(
    """
    <style>

        body{
            margin:0;
            padding:0;
            background:transparent;
            color:white;
            font-family:Arial,sans-serif;
        }

        .hero-container{

            text-align:center;

            padding:32px;

            border-radius:24px;

            background:rgba(
                10,
                22,
                45,
                0.70
            );

            border:1px solid rgba(
                0,
                212,
                255,
                0.22
            );

            box-shadow:
                0px 0px 28px rgba(
                    0,
                    212,
                    255,
                    0.10
                );
        }

        .hero-mini{

            color:#7ad7ff;

            font-size:13px;

            letter-spacing:4px;

            font-weight:800;

            margin-bottom:14px;
        }

        .hero-title{

            color:#00d4ff;

            font-size:44px;

            font-weight:900;

            text-shadow:
                0px 0px 12px #00d4ff;
        }

        .hero-mainline{

            color:white;

            font-size:28px;

            font-weight:900;

            letter-spacing:5px;

            margin-top:12px;

            margin-bottom:12px;
        }

        .hero-subtitle{

            color:#a7d9ff;

            font-size:18px;

            margin-top:10px;
        }

    </style>


    <div class="hero-container">

        <div class="hero-mini">
        CYBER AI SECURITY PLATFORM
        </div>

        <div class="hero-title">
        Adversarial Multimodal AI Framework
        </div>

        <div class="hero-mainline">
        SYNTHETIC PROFILE DETECTION
        </div>

        <div class="hero-subtitle">
        Computer Vision • Transformer NLP • Risk Fusion • Adversarial Intelligence
        </div>

    </div>
    """,

    height=260
)
image_model = ImageDetector(
    "../models/fake_profile_detector.pth"
)

text_model = TextDetector(
    "../models/distilbert_text_model.pth"
)

left_col,right_col = st.columns(
    2
)

with left_col:

    st.markdown(
        "<div class='section-title'>Profile Image Upload</div>",
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader(
        "Upload Profile Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_image:

        st.image(
            uploaded_image,
            use_container_width=True
        )

with right_col:

    st.markdown(
        "<div class='section-title'>Profile Bio Analysis</div>",
        unsafe_allow_html=True
    )

    profile_bio = st.text_area(
        "Enter Profile Bio",
        height=260
    )

analyze_button = st.button(
        "Analyze Profile"
    )    

if analyze_button:

    if uploaded_image and profile_bio:

        with st.spinner(
            "Running Multimodal AI Analysis..."
        ):

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

        tab1,tab2,tab3 = st.tabs(
            [
                "Image Analysis",
                "Text Analysis",
                "Final Assessment"
            ]
        )

        with tab1:

            st.markdown(
                "<div class='cyber-card'>",
                unsafe_allow_html=True
            )

            st.subheader(
                "Computer Vision Module"
            )

            badge_class = (
                "result-fake"
                if image_result["prediction"]=="fake"
                else "result-real"
            )

            st.markdown(
                f"""
                <div class='{badge_class}'>
                {image_result['prediction'].upper()}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                image_result["confidence"]/100
            )

            st.metric(
                "Confidence Score",
                f"{image_result['confidence']}%"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        with tab2:

            st.markdown(
                "<div class='cyber-card'>",
                unsafe_allow_html=True
            )

            st.subheader(
                "Transformer NLP Module"
            )

            badge_class = (
                "result-fake"
                if text_result["prediction"]=="suspicious"
                else "result-real"
            )

            st.markdown(
                f"""
                <div class='{badge_class}'>
                {text_result['prediction'].upper()}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                text_result["confidence"]/100
            )

            st.metric(
                "Confidence Score",
                f"{text_result['confidence']}%"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        with tab3:

            st.markdown(
                "<div class='cyber-card'>",
                unsafe_allow_html=True
            )

            st.subheader(
                "Multimodal Risk Assessment"
            )

            st.metric(
                "Final Verdict",
                fusion_result["verdict"]
            )

            st.metric(
                "Risk Score",
                round(
                    fusion_result["risk_score"],
                    2
                )
            )

            st.progress(
                fusion_result["risk_score"]
            )

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )

            st.subheader(
                "Technical Breakdown"
            )

            col1,col2,col3 = st.columns(
                3
            )

            with col1:

                card_class = (
                    "fake-card"
                    if image_result["prediction"]=="fake"
                    else "real-card"
                )

                st.markdown(
                    f"""
                    <div class="breakdown-card {card_class}">

                    <h3>IMAGE MODULE</h3>

                    Prediction:
                    {image_result['prediction'].upper()}

                    <br><br>

                    Confidence:
                    {image_result['confidence']}%

                    </div>
                    """,

                    unsafe_allow_html=True
                )

            with col2:

                card_class = (
                    "fake-card"
                    if text_result["prediction"]=="suspicious"
                    else "real-card"
                )

                display_prediction = (
                    "FAKE"
                    if text_result["prediction"]=="suspicious"
                    else "REAL"
                )

                st.markdown(
                    f"""
                    <div class="breakdown-card {card_class}">

                    <h3>TEXT MODULE</h3>

                    Prediction:
                    {display_prediction}

                    <br><br>

                    Confidence:
                    {text_result['confidence']}%

                    </div>
                    """,

                    unsafe_allow_html=True
                )

            with col3:

                st.markdown(
                    f"""
                    <div class="breakdown-card verdict-card">

                    <h3>FUSION ENGINE</h3>

                    Verdict:
                    {fusion_result['verdict']}

                    <br><br>

                    Risk Score:
                    {round(fusion_result['risk_score'],2)}

                    </div>
                    """,

                    unsafe_allow_html=True
                )

    else:

        st.warning(
            "Please provide image and profile bio."
        )

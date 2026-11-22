import streamlit as st
from PIL import Image
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Image Classification",
    page_icon="🖼️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background-color: lightblue;
}

[data-testid="stHeading"] h1 {
    color: black;
    text-align: center;
    font-size: 38px;
}

h2, h3 {
    color: black;
}

p, label {
    color: black;
}
.stFileUploader {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 15px;
    border: 2px dashed #A78BFA;
}

.stButton > button {
    background-color: #8052C7;
    color: white;
    border-radius: 10px;
    border: none;
}

.stProgress > div > div > div > div {
    background-color: #8052C7;
}

</style>
""", unsafe_allow_html=True)

# Application title
st.title("🖼️ Image Classification")

st.markdown(
    "<p style='text-align:center;'>"
    "Upload an image and let AI identify its contents."
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# Load pretrained CLIP model
@st.cache_resource
def load_model():
    classifier = pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )
    return classifier

with st.spinner("Loading pretrained CLIP model..."):
    classifier = load_model()

st.success("CLIP model loaded successfully!")

# Image upload
st.subheader("Upload Your Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        width=300
    )

    # Candidate categories
    categories = [
        "a photo of a human hand",
        "a photo of a computer mouse",
        "a photo of a computer keyboard"
    ]

    # Classification
    with st.spinner("Analyzing your image..."):
        results = classifier(
            img,
            candidate_labels=categories
        )

    st.divider()

    st.subheader("✨ Classification Results")

    # Display predictions
    for i, result in enumerate(results):

        label = result["label"]
        confidence = result["score"]

        st.write(
            f"**{i+1}. {label}**"
        )

        st.progress(float(confidence))

        st.caption(
            f"Confidence: {confidence * 100:.2f}%"
        )

    # Final prediction
    st.success(
        f"Predicted Image: {results[0]['label']}"
    )

else:
    st.info("Please upload an image to start classification.")

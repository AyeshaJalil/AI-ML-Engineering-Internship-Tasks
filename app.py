from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_model"


# ---------------------------------------------------------
# Streamlit page settings
# ---------------------------------------------------------

st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="centered",
)


# ---------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    """
    Load the locally saved BERT model and tokenizer.
    The model is loaded only once while the app is running.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The saved_model folder was not found. "
            "Please run train.py and complete model training first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


try:
    tokenizer, model, device = load_model()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

def predict_topic(text: str):
    """
    Predict the news category and probability scores.
    """

    encoded_input = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    encoded_input = {
        key: value.to(device)
        for key, value in encoded_input.items()
    }

    with torch.no_grad():
        output = model(**encoded_input)

    probabilities = torch.softmax(
        output.logits,
        dim=1,
    )[0]

    predicted_id = int(torch.argmax(probabilities).item())

    # Default category names in case the model configuration
    # does not contain the expected labels.
    default_labels = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Science/Technology",
    }

    predicted_label = model.config.id2label.get(
        predicted_id,
        default_labels[predicted_id],
    )

    scores = {}

    for label_id, probability in enumerate(probabilities):
        label_name = model.config.id2label.get(
            label_id,
            default_labels[label_id],
        )

        scores[label_name] = float(probability.item())

    return predicted_label, scores


# ---------------------------------------------------------
# User interface
# ---------------------------------------------------------

st.title("📰 News Topic Classifier")

st.write(
    "This application uses a fine-tuned BERT model to classify "
    "news text into one of four categories."
)

st.info(
    "Available categories: World, Sports, Business, "
    "and Science/Technology."
)

headline = st.text_area(
    "Enter a news headline or short news text:",
    placeholder=(
        "Example: Microsoft launches a new artificial "
        "intelligence platform"
    ),
    height=130,
)

classify_button = st.button(
    "Classify News",
    type="primary",
    use_container_width=True,
)

if classify_button:

    cleaned_headline = headline.strip()

    if not cleaned_headline:
        st.warning("Please enter a news headline first.")

    else:
        with st.spinner("Analyzing the news text..."):
            predicted_topic, category_scores = predict_topic(
                cleaned_headline
            )

        confidence = category_scores[predicted_topic]

        st.success(f"Predicted Topic: {predicted_topic}")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2%}",
        )

        st.subheader("Category probabilities")

        sorted_scores = sorted(
            category_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for category, score in sorted_scores:
            st.write(f"**{category}: {score:.2%}**")
            st.progress(score)

st.divider()

st.caption(
    "Model: bert-base-uncased fine-tuned on the AG News dataset"
)
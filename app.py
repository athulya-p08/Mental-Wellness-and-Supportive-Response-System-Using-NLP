import streamlit as st
import pickle
import random

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🧠 Mental Wellness Support App",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Background Image & CSS
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1470&q=80");
        background-size: cover;
        background-position: center;
        color: #000000;
    }
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.85) !important;
        color: #000000 !important;
        font-size: 18px !important;
        padding: 12px !important;
        border-radius: 12px !important;
    }
    /* TextArea label */
    .stTextArea label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 18px !important;
    }

    /* Placeholder text */
    .stTextArea textarea::placeholder {
        color: #555555 !important;
        opacity: 1 !important;
    }

    /* Warning message box */
    div[data-baseweb="notification"] {
        background-color: rgba(255, 243, 205, 0.95) !important;
        color: #664d03 !important;
        border-radius: 12px !important;
        font-size: 16px !important;
    }

    /* Warning icon color */
    div[data-baseweb="notification"] svg {
        fill: #664d03 !important;
    }

    /* Prevent textarea from fading on button hover */
    .stTextArea textarea:hover,
    .stTextArea textarea:focus,
    .stTextArea textarea:active {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #000000 !important;
        opacity: 1 !important;
    }

    /* 🔒 HARD OVERRIDE: stop Streamlit hover dimming */
    section:hover,
    div:hover,
    main:hover,
    .stApp:hover {
        opacity: 1 !important;
        filter: none !important;
    }

    /* Prevent focus fade caused by buttons */
    .stButton > button:hover ~ div,
    .stButton > button:focus ~ div {
        opacity: 1 !important;
    }

    .stButton>button {
        background-color: #6c5ce7;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        font-size: 18px;
    }

    /* 🔒 Fix Analyze button hover + stop fading */
    .stButton > button {
        background-color: #6c5ce7 !important;
        color: #ffffff !important;
        opacity: 1 !important;
    }

    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active {
        background-color: #5a4bdc !important;
        color: #ffffff !important;
        opacity: 1 !important;
        box-shadow: 0 0 0 2px rgba(108, 92, 231, 0.4);
    }

    /* Prevent button hover from dimming other widgets */
    .stButton:hover ~ *,
    .stButton:focus ~ * {
        opacity: 1 !important;
        filter: none !important;
    }

    .response-box {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
with open("mental_wellness_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# Supportive Responses
# -----------------------------
responses = {
    "stress": [
        "You're under a lot of pressure, and that can be exhausting.",
        "Try breaking tasks into small steps — you don't have to do everything at once.",
        "Short breaks and rest can actually help you regain focus.",
        "Remember to breathe deeply — your mental health matters."
    ],
    "anxiety": [
        "It sounds like anxiety is weighing on you. Try taking slow, deep breaths to calm your nervous system.",
        "Anxiety can make everything feel urgent, but you're safe right now.",
        "You're doing your best, and that's enough for this moment.",
        "Focus on what you can control and let go of what you can't."
    ],
    "sadness": [
        "I'm really sorry you're feeling this way. You’re not alone.",
        "Give yourself time — it’s okay to feel sad sometimes.",
        "Reach out to someone you trust, talking helps.",
        "Allow yourself to feel and process your emotions."
    ],
    "anger": [
        "Strong emotions are natural, especially when things feel unfair.",
        "Pausing before reacting can help prevent regret later.",
        "Taking a few deep breaths may help release some of this tension.",
        "Consider journaling your feelings or taking a walk to calm down."
    ],
    "fear": [
        "Feeling afraid can be unsettling, but this feeling will not last forever.",
        "You're stronger than you think, even when fear says otherwise.",
        "Try grounding yourself in the present moment.",
        "Focus on your breath and what you can control right now."
    ],
    "depression": [
        "It sounds really heavy. You deserve care and understanding.",
        "Reach out to someone you trust — even small steps help.",
        "Focus on one small task at a time; it’s okay to rest.",
        "Take gentle care of yourself and don’t hesitate to ask for help."
    ],
    "loneliness": [
        "Feeling lonely can be painful, but it doesn't mean you're unwanted.",
        "Even small connections can make a difference.",
        "You deserve companionship and understanding.",
        "Reach out to a friend, family member, or online community."
    ],
    "happiness": [
        "Yay! That’s wonderful — cherish these joyful moments.",
        "Celebrate your happiness and share it with others if you can.",
        "Keep enjoying the little things that make you smile.",
        "Your positive emotions are important — savor them fully."
    ],
    "neutral": [
        "Thank you for sharing how you feel.",
        "It's good that you're checking in with yourself.",
        "Keep taking care of your mental well-being."
    ],
    "default": [
        "Thank you for opening up. Your feelings matter.",
        "You're not alone — support is always available.",
        "It's okay to ask for help when things feel heavy."
    ]
}

# -----------------------------
# App UI
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#6c5ce7;'>
    🧠 Mental Wellness Support App
    </h1>
    <p style='text-align:center; font-size:20px; color:#000000;'>
    Share how you're feeling, and receive a supportive response 💙
    </p>
    """,
    unsafe_allow_html=True
)

user_input = st.text_area(
    "How do you feel today?",
    placeholder="Example: I feel bad because I failed my exam..."
)

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Analyze your emotion"):
    if user_input.strip() == "":
        st.warning("Please enter how you are feeling.")
    else:
        input_vector = vectorizer.transform([user_input])
        emotion = model.predict(input_vector)[0].lower()

        # Pick multiple responses
        emotion_responses = responses.get(emotion, responses["default"])
        selected_responses = random.sample(emotion_responses, min(4, len(emotion_responses)))

        # Color & emoji mapping
        color_map = {
            "happiness": "#00C853", "sadness": "#2979FF", "stress": "#FFD600",
            "anxiety": "#FF6D00", "anger": "#D50000", "fear": "#9C27B0",
            "depression": "#6A1B9A", "loneliness": "#009688", "neutral": "#607D8B"
        }
        emoji_map = {
            "happiness": "🥳", "sadness": "😢", "stress": "😓", "anxiety": "😰",
            "anger": "😡", "fear": "😨", "depression": "😔", "loneliness": "😞",
            "neutral": "😐"
        }

        # -----------------------------
        # Output UI
        # -----------------------------
        st.markdown("---")
        st.subheader("🧠 Detected Emotion")
        st.markdown(
            f"<div style='background-color:{color_map.get(emotion,'#FFFFFF')}; "
            f"padding:15px; border-radius:12px; font-weight:bold; font-size:20px; color:#000000;'>"
            f"{emoji_map.get(emotion,'')} {emotion.capitalize()}</div>",
            unsafe_allow_html=True
        )

        st.subheader("💬 Supportive Message")
        for line in selected_responses:
            st.markdown(
                f"<div class='response-box'>• {line}</div>",
                unsafe_allow_html=True
            )

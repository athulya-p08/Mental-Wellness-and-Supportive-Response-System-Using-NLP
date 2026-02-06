import pickle
import re
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (runs once, then cached)
nltk.download('punkt')
nltk.download('stopwords')

# Load trained model
with open("mental_wellness_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

# Text cleaning function (same logic as preprocess)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Emotion prediction function
def predict_emotion(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction.lower()  # <-- convert to lowercase for dictionary lookup

# Supportive responses (MULTIPLE per emotion)
responses = {
    "stress": [
        "It sounds like you're under a lot of pressure. Taking short breaks and prioritizing tasks may help.",
        "Focus on one task at a time to reduce overwhelm.",
        "Remember to breathe deeply — your mental health matters."
    ],
    "anxiety": [
        "Feeling anxious can be overwhelming. Try slow breathing or grounding techniques to calm your mind.",
        "You are safe in this moment — focus on your senses and take small steps.",
        "Remember, thoughts are not facts. Pause and breathe."
    ],
    "sadness": [
        "I’m really sorry you’re feeling this way. You’re not alone.",
        "Give yourself time — it’s okay to feel sad sometimes.",
        "Reach out to someone you trust, talking helps."
    ],
    "anger": [
        "Strong emotions are natural. Pausing, breathing deeply, or stepping away briefly can help.",
        "Consider journaling your feelings or taking a walk to release tension.",
        "Anger is valid, but notice it and respond mindfully."
    ],
    "fear": [
        "Feeling afraid can be unsettling. Remind yourself this moment will pass.",
        "Focus on your breath and what you can control right now.",
        "It’s okay to feel scared — safety is in the present moment."
    ],
    "depression": [
        "It sounds really heavy. You deserve care and understanding.",
        "Reach out to someone you trust — even small steps help.",
        "Focus on one small task at a time; it’s okay to rest."
    ],
    "loneliness": [
        "Feeling lonely can be painful. Small connections, even brief ones, help.",
        "Reach out to a friend or family member — connection matters.",
        "You’re not alone; online communities or helplines can help."
    ],
    "happiness": [
        "Yay! That’s wonderful — cherish these joyful moments.",
        "Celebrate your happiness and share it with others if you can.",
        "Keep enjoying the little things that make you smile.",
        "Your positive emotions are important — savor them fully."
    ],
    "neutral": [
        "Thanks for sharing how you feel. Keep taking care of your mental well-being."
    ]
}

# Main execution
if __name__ == "__main__":
    user_input = input("💭 How do you feel today? ")

    emotion = predict_emotion(user_input)

    print("\n🧠 Detected Emotion:", emotion.capitalize())
    print("\n💬 Supportive Message:")

    # Randomly pick 3 messages if available
    possible_responses = responses.get(emotion, ["You are not alone."])
    k = min(3, len(possible_responses))
    for line in random.sample(possible_responses, k):
        print("•", line)

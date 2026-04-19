import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- DATA ----------------
df = pd.read_csv("C:/Users/Bhavn/OneDrive/Desktop/AI_Food_Project/zomato.csv")

df = df.dropna()

df["features"] = (
    df["Cuisine"].astype(str) + " " +
    df["Place_Name"].astype(str)
)

# ---------------- VECTOR ----------------
vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(df["features"])

# ---------------- AI SEARCH ENGINE ----------------
def google_search(query):

    query = query.lower()

    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, matrix)[0]

    results = []

    for i in range(len(df)):
        sim = sim_scores[i]
        rating = df.iloc[i]["Dining_Rating"]
        cost = df.iloc[i]["Prices"]

        # SMART AI RANKING
        score = (
            sim * 0.6 +
            (rating / 5) * 0.3 -
            (cost / 1000) * 0.1
        )

        results.append((i, score))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    top_indices = [r[0] for r in results[:5]]

    return df.iloc[top_indices]

# ---------------- SESSION CHAT MEMORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- UI ----------------
st.set_page_config(page_title="AI Restaurant Chatbot", layout="centered")

st.title("🤖 AI Restaurant Chatbot (Google + ChatGPT Style)")

st.markdown("Ask me anything like: *cheap indian food in Delhi* 🍽️")

# ---------------- SHOW CHAT HISTORY ----------------
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["text"])

# ---------------- INPUT ----------------
query = st.chat_input("Type your food search...")

# ---------------- PROCESS ----------------
if query:

    # USER MESSAGE
    st.session_state.chat_history.append({
        "role": "user",
        "text": query
    })

    results = google_search(query)

    # RESPONSE
    if results.empty:
        response = "😢 No restaurants found"
    else:
        response = ""

        for _, r in results.iterrows():
            response += (
                f"🍽️ **{r['Restaurant_Name']}**\n"
                f"⭐ Rating: {r['Dining_Rating']} | 💰 {r['Prices']}\n"
                f"📍 {r['Place_Name']} | 🍛 {r['Cuisine']}\n\n"
            )

    # ASSISTANT MESSAGE
    st.session_state.chat_history.append({
        "role": "assistant",
        "text": response
    })

    st.rerun()

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import numpy as np

st.set_page_config(page_title="Hotel Review Analyzer", page_icon="🏨", layout="centered")

@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={"reviewText": "text", "Positive": "label"})
    df = df.sample(2000, random_state=42).reset_index(drop=True)
    
    positive = df[df["label"]==1]
    negative = df[df["label"]==0]
    negative_up = resample(negative, replace=True, n_samples=len(positive), random_state=42)
    balanced = pd.concat([positive, negative_up]).sample(frac=1, random_state=42)
    
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    X = tfidf.fit_transform(balanced["text"])
    y = balanced["label"]
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    return model, tfidf

model, tfidf = train_model()

st.title("Hotel Review Sentiment Analyzer")
st.markdown("Powered by Machine Learning — paste any hotel review below to analyze its sentiment.")

st.divider()

review_input = st.text_area("Enter a hotel review:", height=150,
    placeholder="e.g. The room was fantastic, staff were incredibly helpful...")

col1, col2 = st.columns([1, 3])

with col1:
    analyze_btn = st.button("Analyze", type="primary")

if analyze_btn and review_input:
    vec = tfidf.transform([review_input])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = prob[pred] * 100
    
    st.divider()
    
    if pred == 1:
        st.success(f"Sentiment: {sentiment}")
    else:
        st.error(f"Sentiment: {sentiment}")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Positive probability", f"{prob[1]*100:.1f}%")
    with col_b:
        st.metric("Negative probability", f"{prob[0]*100:.1f}%")
    
    st.progress(float(prob[1]))
    st.caption(f"Confidence: {confidence:.1f}%")

st.divider()
st.subheader("Try these examples:")

examples = [
    "The room was absolutely fantastic, staff were incredibly helpful and friendly!",
    "Terrible experience, dirty room, rude staff, never coming back to this hotel.",
    "Outstanding service, beautiful pool, perfect location near everything.",
    "Broken AC, noisy neighbors, overpriced breakfast. Very disappointed."
]

for example in examples:
    if st.button(example[:60] + "...", key=example):
        vec = tfidf.transform([example])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        if pred == 1:
            st.success(f"{sentiment} — {prob[1]*100:.1f}% confidence")
        else:
            st.error(f"{sentiment} — {prob[0]*100:.1f}% confidence")
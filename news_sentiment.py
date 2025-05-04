import os
import certifi
from dotenv import load_dotenv
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

import requests
import streamlit as st
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from textblob import TextBlob
from collections import Counter
from datetime import datetime, timedelta

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Sentiment + Summarization ===

def get_sentiment(text: str) -> str:
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "POSITIVE"
        elif polarity < -0.1:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    except:
        return "NEUTRAL"

def gpt_summarize(text: str, max_sentences: int = 3) -> str:
    try:
        messages = [
            {"role": "system", "content": "Summarize this weather article in 2–3 concise sentences:"},
            {"role": "user",   "content": text[:2000]}
        ]
        return client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        ).choices[0].message.content.strip()
    except:
        return text[:300] + "..."

# === Helpers ===

def clean_snippet(snippet: str) -> str:
    if snippet:
        soup = BeautifulSoup(snippet, "html.parser")
        return soup.get_text().strip()
    return "No summary available."

def scrape_article_text(url: str) -> str:
    try:
        page = requests.get(url, timeout=5)
        soup = BeautifulSoup(page.content, "html.parser")
        paragraphs = soup.find_all('p')
        return " ".join(p.get_text() for p in paragraphs).strip()
    except:
        return ""

def filter_by_time(results: list[dict], time_range: str) -> list[dict]:
    if time_range == "Anytime":
        return results

    now = datetime.now()
    if time_range == "This week":
        threshold = now - timedelta(days=7)
    elif time_range == "This month":
        threshold = now - timedelta(days=30)
    elif time_range == "This year":
        threshold = now - timedelta(days=365)
    else:
        return results

    def within_range(pub_date: str) -> bool:
        try:
            dt = datetime.strptime(pub_date[:10], "%Y-%m-%d")
            return dt >= threshold
        except:
            return True  # include if missing or malformed

    filtered = [r for r in results if within_range(r.get("date", ""))]
    return filtered or results  # fallback to unfiltered if empty

# === Plotting ===

def plot_sentiment_distribution(sentiments: list[str]) -> None:
    base = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
    counts = {**base, **Counter(sentiments)}
    labels = list(counts.keys())
    values = list(counts.values())

    col1, col2 = st.columns(2)
    with col1:
        fig_bar = go.Figure([go.Bar(x=labels, y=values)])
        fig_bar.update_layout(
            title="🧠 Sentiment Distribution – Bar Chart",
            xaxis_title="Sentiment",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        fig_pie = go.Figure([go.Pie(labels=labels, values=values, hole=0.4)])
        fig_pie.update_layout(title="🧠 Sentiment Distribution – Pie Chart")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.caption("📰 Source: Google News via SerpAPI | NLP via TextBlob + GPT")

# === MAIN ENTRY ===

def get_weather_news_and_sentiment(city: str) -> None:
    st.markdown(f"### 📰 News Summary – {city.title()}")

    # UI inputs: default to 5 articles, default time range = This week
    num_articles = st.selectbox(
        "🔢 How many articles to fetch?",
        [5, 10, 15, 20],
        index=0
    )
    time_range = st.selectbox(
        "🕒 Time Range",
        ["Anytime", "This week", "This month", "This year"],
        index=1
    )

    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        st.error("❌ SERPAPI_KEY is missing.")
        return

    # fetch more to ensure enough relevant ones, sorted by date
    params = {
        "q":             f"{city} weather forecast OR climate update",
        "location":      city,
        "api_key":       api_key,
        "engine":        "google_news",
        "num":           num_articles * 5,
        "sort_by_date":  "true"
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        st.error("❌ Failed to fetch news.")
        return

    results = response.json().get("news_results", [])
    if not results:
        st.info("📰 No relevant news found.")
        return

    filtered = filter_by_time(results, time_range)
    weather_keywords = [
        "weather", "forecast", "rain", "climate", "temperature",
        "humidity", "monsoon", "heat", "cold"
    ]

    sentiments = []
    shown = 0

    for article in filtered:
        if shown >= num_articles:
            break

        title   = article.get("title", "")
        snippet = clean_snippet(article.get("snippet", ""))
        link    = article.get("link")
        source  = article.get("source", {}).get("name", "Unknown")
        text    = scrape_article_text(link)

        content = (title + snippet + text).lower()
        if not any(k in content for k in weather_keywords):
            continue

        summary   = gpt_summarize(text or snippet)
        sentiment = get_sentiment(summary)

        st.markdown(
            f"**📰 {title}** ({source})  \n"
            f"{summary}  \n"
            f"[Read more]({link})  \n"
            f"_Sentiment: **{sentiment.title()}**_\n"
        )

        sentiments.append(sentiment)
        shown += 1

    if shown == 0:
        st.info("📰 No weather‑relevant articles found in the selected range.")
        return

    # plot exactly those shown
    plot_sentiment_distribution(sentiments)
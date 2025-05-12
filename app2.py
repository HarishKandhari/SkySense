# app2.py
# AI Weather Chatbot using LLM and RAG

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import re
import json
import pickle
import requests
import streamlit as st
from datetime import datetime, timedelta
from openai import OpenAI
import logging
import os
from uuid import uuid4
from meteostat import Point, Daily, Stations
import pandas as pd

from utils import CITY_COORDINATES, get_direct_weather_answer
from live_weather import (
    get_current_weather, get_air_quality, get_uv_index, get_forecast,
    get_weekly_forecast, get_yesterday_windspeed,
    get_yesterday_temperature, get_max_temp_last_year
)
from historical_analysis import is_historical_query, analyze_historical_query
from news_sentiment import get_weather_news_and_sentiment
from visuals import plot_weather_metrics, plot_monthly_trends
from compare import compare_weather
from retriever import search_similar_docs
from rag_engine import get_rag_response

# ─── SETUP LOGGING ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── LOAD ENV & KEYS ─────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Check for missing API keys
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY is missing")
    st.error("⚠️ OpenAI API key is missing. Please set it in the environment.")
    st.stop()
if not OPENWEATHER_API_KEY:
    logging.warning("OPENWEATHER_API_KEY is missing; weather data may be unavailable")

client = OpenAI(api_key=OPENAI_API_KEY)

# ─── LOAD FAISS INDEX ────────────────────────────────────────────────────────
VEC_PATH = "data/vector_index.pkl"
try:
    with open(VEC_PATH, "rb") as f:
        index, texts = pickle.load(f)
except Exception as e:
    logging.error(f"Failed to load FAISS index: {e}")
    st.error("⚠️ Vector index not found. Please run build_index.py.")
    st.stop()

# ─── HISTORICAL DATA HELPER ──────────────────────────────────────────────────
def get_last_rain_event(city: str) -> str:
    """
    Fetch the most recent date with precipitation for the given city using Meteostat.
    Tries up to 10 nearby stations and a 180-day lookback.
    Returns a formatted string with the date or an error message.
    """
    try:
        lat, lon = CITY_COORDINATES.get(city.lower(), (None, None))
        if not lat or not lon:
            logging.error(f"No coordinates found for {city}")
            return f"⚠️ Coordinates not found for {city.title()}."

        # Try primary station
        end = datetime.now()
        start = end - timedelta(days=180)  # Check last 180 days
        point = Point(lat, lon)
        data = Daily(point, start, end).fetch()
        
        if not data.empty and "prcp" in data.columns:
            data = data[data["prcp"] > 0.1]  # Significant precipitation (>0.1 mm)
            if not data.empty:
                last_rain_date = data.index[-1].strftime("%Y-%m-%d")
                prcp_amount = data["prcp"].iloc[-1]
                logging.info(f"Found precipitation for {city} on {last_rain_date}: {prcp_amount} mm")
                return f"The last significant rain in {city.title()} was on {last_rain_date} with {prcp_amount:.1f} mm of precipitation."

        # Try up to 10 nearby stations
        stations = Stations()
        stations = stations.nearby(lat, lon).fetch(10)  # Get up to 10 nearby stations
        for station_id in stations.index:
            try:
                data = Daily(station_id, start, end).fetch()
                if not data.empty and "prcp" in data.columns:
                    data = data[data["prcp"] > 0.1]
                    if not data.empty:
                        last_rain_date = data.index[-1].strftime("%Y-%m-%d")
                        prcp_amount = data["prcp"].iloc[-1]
                        logging.info(f"Found precipitation for {city} at station {station_id} on {last_rain_date}: {prcp_amount} mm")
                        return f"The last significant rain in {city.title()} (nearby station) was on {last_rain_date} with {prcp_amount:.1f} mm of precipitation."
            except Exception as e:
                logging.warning(f"Failed to fetch data for station {station_id} near {city}: {e}")
                continue

        logging.error(f"No precipitation data found for {city} in the last 180 days")
        return f"⚠️ No precipitation data available for {city.title()} in the last 180 days from nearby stations. Please check Meteostat data availability."
    except Exception as e:
        logging.error(f"Failed to fetch last rain event for {city}: {e}")
        return f"⚠️ Unable to fetch historical precipitation data for {city.title()}. Please check Meteostat data availability or network connection."

# ─── FORECAST HELPERS ────────────────────────────────────────────────────────
def fetch_5day_json(city: str) -> dict:
    """
    Fetch the 5-day/3-hour forecast for the given city using OpenWeather API.
    """
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Failed to fetch 5-day forecast for {city}: {e}")
        return {}

def will_rain_on(city: str, offset_days: int) -> bool:
    """
    Check if rain is likely for the given city and day offset using OpenWeather API.
    Returns True if rain is likely (pop > 0.3 or rain/shower in conditions).
    """
    data = fetch_5day_json(city)
    if "list" not in data or not data["list"]:
        logging.error(f"No forecast data available for {city}")
        return False
    target = (datetime.utcnow() + timedelta(days=offset_days)).date().isoformat()
    for e in data["list"]:
        if e["dt_txt"].startswith(target):
            main = e["weather"][0]["main"].lower()
            pop = e.get("pop", 0)
            if pop > 0.3 or "rain" in main or "shower" in main:
                return True
    return False

def get_max_temp_today(city: str, offset_days: int = 0) -> str:
    """
    Fetch the highest temperature for the given city and day offset using OpenWeather API.
    Returns a formatted string with the max temperature or an error message.
    """
    try:
        data = fetch_5day_json(city)
        if "list" not in data or not data["list"]:
            logging.error(f"No forecast data available for {city}")
            return "⚠️ Forecast data not available."
        
        target = (datetime.utcnow() + timedelta(days=offset_days)).date().isoformat()
        daily_temps = [
            e["main"]["temp_max"]
            for e in data["list"]
            if e["dt_txt"].startswith(target)
        ]
        
        if not daily_temps:
            logging.error(f"No forecast data for {city} on {target}")
            return f"⚠️ No forecast data available for {target}."
        
        max_temp = max(daily_temps)
        # Get conditions for the time slot with the max temperature
        max_entry = max(
            [e for e in data["list"] if e["dt_txt"].startswith(target)],
            key=lambda e: e["main"]["temp_max"]
        )
        desc = max_entry["weather"][0]["description"].capitalize()
        pop = max_entry.get("pop", 0) * 100
        return f"{max_temp:.2f}°C, {desc}, {pop:.0f}% chance of precipitation"
    except Exception as e:
        logging.error(f"Failed to fetch max temperature for {city}: {e}")
        return f"⚠️ Unable to fetch forecast data for {city.title()}. Please check OpenWeather API key."

def summary_on(city: str, offset_days: int) -> str:
    """
    Fetch the forecast summary for the given city and day offset using OpenWeather API.
    Returns a formatted string with conditions, temperature, and precipitation chance.
    """
    try:
        data = fetch_5day_json(city)
        if "list" not in data or not data["list"]:
            logging.error(f"No forecast data available for {city}")
            return "⚠️ Forecast data not available."
        
        target = (datetime.utcnow() + timedelta(days=offset_days)).date().isoformat()
        cands = [e for e in data["list"] if e["dt_txt"].startswith(target)]
        if not cands:
            logging.error(f"No forecast data for {city} on {target}")
            return f"⚠️ No forecast for {target}."
        
        # Select entry closest to noon for general conditions
        entry = min(cands, key=lambda e: abs(int(e["dt_txt"][11:13]) - 12))
        desc = entry["weather"][0]["description"].capitalize()
        temp = entry["main"]["temp"]
        pop = entry.get("pop", 0) * 100
        return f"{desc}, {temp:.2f}°C, {pop:.0f}% chance of precipitation"
    except Exception as e:
        logging.error(f"Failed to fetch forecast summary for {city}: {e}")
        return f"⚠️ Unable to fetch forecast data for {city.title()}. Please check OpenWeather API key."

def get_weekend_forecast(city: str) -> str:
    """
    Fetch the weekend forecast for the given city using OpenWeather API.
    Returns a formatted string with conditions for Saturday and Sunday.
    """
    try:
        data = fetch_5day_json(city)
        if "list" not in data or not data["list"]:
            logging.error(f"No forecast data available for {city}")
            return "⚠️ Weekend forecast data not available."
        
        today = datetime.utcnow().weekday()
        days_to_saturday = (5 - today) % 7
        days_to_sunday = (6 - today) % 7
        saturday = (datetime.utcnow() + timedelta(days=days_to_saturday)).date().isoformat()
        sunday = (datetime.utcnow() + timedelta(days=days_to_sunday)).date().isoformat()
        forecast = []
        for e in data["list"]:
            date = e["dt_txt"][:10]
            if date in (saturday, sunday) and e["dt_txt"][11:13] == "12":
                desc = e["weather"][0]["description"].capitalize()
                temp = e["main"]["temp"]
                pop = e["pop"] * 100
                forecast.append(f"- **{date}**: {desc}, {temp:.2f}°C, {pop:.0f}% chance of precipitation")
        if not forecast:
            logging.error(f"No weekend forecast data for {city}")
            return "⚠️ No weekend forecast available."
        return f"**{city.title()} Weekend Forecast**\n" + "\n".join(forecast)
    except Exception as e:
        logging.error(f"Failed to fetch weekend forecast for {city}: {e}")
        return f"⚠️ Unable to fetch weekend forecast for {city.title()}. Please check OpenWeather API key."

# ─── LLM RESPONSE GENERATOR ──────────────────────────────────────────────────
def generate_llm_response(query: str, context: str, city: str = None, forecast_data: str = None) -> str:
    """
    Generate a response using the LLM with RAG context, city-specific weather data, and forecast data.
    """
    system_prompt = """
You are SkySense, a friendly and expert weather assistant for Indian and global weather.
Use the provided context, weather data, and forecast data to answer the user's query accurately and conversationally.
Focus on the forecast or historical data for precise predictions and avoid speculative statements.
Do not say you lack forecast or historical data if it is provided.
Incorporate RAG context for additional climate or historical insights, such as past monsoon patterns or recent weather events.
Cite facts where possible and avoid hallucination.
Respond in a concise, natural tone, using markdown for formatting.
If the query is unclear or lacks a city, politely ask for clarification and list supported cities (Delhi, Mumbai, Bangalore, Kolkata, Chennai, Hyderabad).
"""
    weather_data = ""
    if city:
        try:
            ico, prev = get_current_weather(city)
            weather_data = f"**Current Weather in {city.title()}**:\n{prev}\n"
        except Exception as e:
            logging.error(f"Failed to fetch weather for {city}: {e}")
            weather_data = f"⚠️ Unable to fetch weather data for {city.title()}. Please ensure the OpenWeather API key is set.\n"

    user_prompt = f"""
**Query**: {query}
**Weather Data** (if applicable): {weather_data}
**Forecast Data** (if applicable): {forecast_data or 'None'}
**Context** (from RAG): {context}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"LLM response generation failed: {e}")
        return "⚠️ Sorry, I couldn’t process your query. Please check the OpenAI API key and try again."

# ─── INTENT PARSER ───────────────────────────────────────────────────────────
def parse_user_intent(query: str, last_city: str = None) -> dict:
    """
    Parse user query to determine intent and extract city using LLM.
    Returns: {"intent": str, "city": str or None, "cities": list}
    """
    system_prompt = """
You are a weather assistant intent parser. Analyze the user query and output JSON with:
- intent: One of ["greeting", "weather_query", "activity_query", "historical_query", "compare_query", "map_query", "general_query"]
- city: Extracted city name (lowercase) or null
- cities: List of up to two city names (lowercase) for comparison queries
Return only valid JSON.
"""
    user_prompt = f"Query: {query}\nLast known city: {last_city or 'None'}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logging.error(f"Intent parsing failed: {e}")
        return {"intent": "general_query", "city": None, "cities": []}

# ─── SETUP STREAMLIT ─────────────────────────────────────────────────────────
st.set_page_config(page_title="SkySense", layout="wide")
st.title("🌤️ SkySense – AI‑Powered Weather & Climate Chatbot")

state = st.session_state
state.setdefault("history", [])
state.setdefault("last_city", None)
state.setdefault("pending_query", None)
state.setdefault("view_switch", "📅 Forecast")
state.setdefault("mode", "💬 Chat Mode")

KNOWN_CITIES = ["delhi", "mumbai", "bangalore", "kolkata", "chennai", "hyderabad"]

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🌍 Weather Options")
    state.mode = st.radio("Mode", ["💬 Chat Mode", "📊 Tool Mode"],
                         index=0 if state.mode == "💬 Chat Mode" else 1)
    choice = st.selectbox("City", [c.title() for c in KNOWN_CITIES],
                          index=KNOWN_CITIES.index(state.last_city) if state.last_city in KNOWN_CITIES else 0)
    state.last_city = choice.lower()
    try:
        ico, prev = get_current_weather(state.last_city)
        st.image(ico, width=60)
        st.markdown(prev)
    except Exception as e:
        logging.error(f"Sidebar weather fetch failed: {e}")
        st.markdown("⚠️ Weather not available. Please check the OpenWeather API key.")

# ─── SHOW HISTORY ─────────────────────────────────────────────────────────────
for msg in state.history:
    st.chat_message(msg["role"]).markdown(msg["content"], unsafe_allow_html=True)

# ─── TOOL MODE ───────────────────────────────────────────────────────────────
if state.mode == "📊 Tool Mode":
    logging.info(f"Rendering Tool Mode view: {state.view_switch} for city: {state.last_city}")
    st.subheader(f"{state.view_switch} – {state.last_city.title()}")
    try:
        if state.view_switch == "📅 Forecast":
            forecast = get_forecast(state.last_city)
            if forecast.startswith("⚠️"):
                st.markdown(forecast)
            else:
                st.markdown(forecast, unsafe_allow_html=True)
                lines = [
                    f"- **{(datetime.utcnow() + timedelta(days=i)).date()}:** {summary_on(state.last_city, i)}"
                    for i in range(1, 6)
                ]
                #st.markdown("### 📖 Forecast Summary\n" + "\n".join(lines))
        elif state.view_switch == "📈 Monthly Trends":
            plot_monthly_trends(state.last_city)
        elif state.view_switch == "📰 News Summary":
            get_weather_news_and_sentiment(state.last_city)
        elif state.view_switch == "📊 Charts":
            plot_weather_metrics(state.last_city)
        elif state.view_switch == "🏭 Air Quality":
            st.markdown(get_air_quality(state.last_city))
        elif state.view_switch == "🌞 UV Index":
            st.markdown(get_uv_index(state.last_city))
        elif state.view_switch == "🌐 Compare Cities":
            sel = st.multiselect("Choose 2+ cities", [c.title() for c in KNOWN_CITIES],
                                 default=[state.last_city.title()])
            if len(sel) >= 2:
                compare_weather(sel)
            else:
                st.markdown("ℹ️ Please select at least two cities to compare.")
    except Exception as e:
        logging.error(f"Tool Mode view '{state.view_switch}' failed: {e}")
        st.markdown(f"⚠️ Unable to display {state.view_switch}. Please check the relevant API key (OpenWeather, SerpAPI) and try again.")

# ─── CHAT INPUT & LOGIC ──────────────────────────────────────────────────────
if user_input := st.chat_input("Ask me anything about the weather…"):
    st.chat_message("user").markdown(user_input)
    state.history.append({"role": "user", "content": user_input})
    query = user_input.strip()
    query_lower = query.lower()

    # Parse intent using LLM
    with st.spinner("☁️ Analyzing your query…"):
        parsed = parse_user_intent(query, state.last_city)
    intent = parsed.get("intent", "general_query")
    city = parsed.get("city")
    cities = parsed.get("cities", [])

    logging.info(f"Parsed intent: {intent}, city: {city}, cities: {cities}")

    # Update last city if a new one is detected
    if city and city in KNOWN_CITIES:
        state.last_city = city
    elif city and city not in KNOWN_CITIES:
        response = f"⚠️ Sorry, I only support {', '.join(c.title() for c in KNOWN_CITIES)}. Please choose one or ask about another topic."
        st.chat_message("assistant").markdown(response)
        state.history.append({"role": "assistant", "content": response})
        st.stop()

    # Handle specific intents
    if intent == "greeting":
        with st.spinner("☁️ Retrieving relevant information…"):
            try:
                docs = search_similar_docs(query, index, texts, k=3)
                rag_context = "\n\n".join(docs)
                logging.info(f"RAG context for query '{query}': {rag_context}")
            except Exception as e:
                logging.error(f"RAG search failed: {e}")
                rag_context = "No relevant documents found."
        response = generate_llm_response(query, rag_context, None)
        st.chat_message("assistant").markdown(response)
        state.history.append({"role": "assistant", "content": response})
        st.stop()

    elif intent == "map_query":
        place = city or state.last_city
        if not place:
            response = f"🌍 Please tell me which place you'd like to see the map of. Supported cities: {', '.join(c.title() for c in KNOWN_CITIES)}."
            st.chat_message("assistant").markdown(response)
            state.history.append({"role": "assistant", "content": response})
            st.stop()
        if not MAPBOX_API_KEY:
            coords = CITY_COORDINATES.get(place.lower())
            response = (
                f"⚠️ Mapbox API key is missing, so I can’t display a map of {place.title()}. "
                f"Please set the MAPBOX_API_KEY in your .env file to view maps.\n\n"
                f"As a fallback, here are the coordinates for {place.title()}: "
                f"Latitude {coords[0]}, Longitude {coords[1]}."
            ) if coords else (
                f"⚠️ Mapbox API key is missing, so I can’t display a map of {place.title()}. "
                f"Please set the MAPBOX_API_KEY in your .env file to view maps."
            )
            st.chat_message("assistant").markdown(response)
            state.history.append({"role": "assistant", "content": response})
            st.stop()
        try:
            geo_resp = requests.get(
                f"https://api.mapbox.com/geocoding/v5/mapbox.places/{requests.utils.quote(place)}.json"
                f"?access_token={MAPBOX_API_KEY}&limit=1"
            ).json()
            features = geo_resp.get("features", [])
            if not features:
                response = f"⚠️ Could not find '{place.title()}'."
                st.chat_message("assistant").markdown(response)
                state.history.append({"role": "assistant", "content": response})
                st.stop()
            lon, lat = features[0]["center"]
            map_url = (
                f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/"
                f"static/{lon},{lat},10,0/600x400"
                f"?access_token={MAPBOX_API_KEY}"
            )
            st.chat_message("assistant").image(map_url, caption=f"Map of {place.title()}")
            state.history.append({"role": "assistant", "content": f"Displayed map of {place.title()}."})
            st.stop()
        except Exception as e:
            logging.error(f"Map request failed: {e}")
            response = "⚠️ Unable to fetch map at this time. Please check the Mapbox API key."
            st.chat_message("assistant").markdown(response)
            state.history.append({"role": "assistant", "content": response})
            st.stop()

    elif intent == "compare_query" and len(cities) >= 2:
        try:
            compare_weather([c.title() for c in cities])
            context = f"Comparison requested for {', '.join(c.title() for c in cities)}. Weather data has been plotted."
            response = generate_llm_response(query, context, city)
            st.chat_message("assistant").markdown(response)
            state.history.append({"role": "assistant", "content": response})
            st.stop()
        except Exception as e:
            logging.error(f"City comparison failed: {e}")
            response = "⚠️ Unable to compare cities at this time. Please check the OpenWeather API key."
            st.chat_message("assistant").markdown(response)
            state.history.append({"role": "assistant", "content": response})
            st.stop()

    elif intent == "historical_query":
        try:
            # Check for "last rain" or "recent rain" queries
            if ("last rain" in query_lower or "recently rain" in query_lower or "did it rain" in query_lower) and city:
                historical_response = get_last_rain_event(city)
                if historical_response.startswith("⚠️"):
                    # Fallback to RAG context and X post data
                    docs = search_similar_docs(query, index, texts, k=3)
                    rag_context = "\n\n".join(docs)
                    logging.info(f"RAG context for last rain query '{query}': {rag_context}")
                    fallback_context = rag_context
                    if city.lower() == "bangalore":
                        fallback_context += "\nRecent reports indicate heavy rain in Bengaluru on May 1, 2025, with thunder and lightning."
                    elif city.lower() == "chennai":
                        fallback_context += "\nOur knowledge base notes urban flooding in Chennai in December 2019 due to northeast monsoon rains, suggesting recent similar events."
                    elif city.lower() == "delhi":
                        fallback_context += "\nRecent reports indicate moderate to heavy showers in Delhi NCR on May 1, 2025, following a dust storm."
                    historical_response = generate_llm_response(
                        query, fallback_context, city,
                        forecast_data="No recent precipitation data available; using historical context and recent reports."
                    )
                else:
                    docs = search_similar_docs(query, index, texts, k=3)
                    rag_context = "\n\n".join(docs)
                    logging.info(f"RAG context for last rain query '{query}': {rag_context}")
            elif "highest temperature" in query_lower and city:
                max_temp_response = get_max_temp_last_year(city)
                if not max_temp_response.startswith("⚠️"):
                    historical_response = max_temp_response
                else:
                    if city.lower() == "delhi":
                        historical_response = (
                            "The highest temperature in Delhi last year (2024) was 49.9°C, "
                            "recorded on May 28 and 29 at Mungeshpur, according to the India Meteorological Department."
                        )
                    else:
                        historical_response = max_temp_response
                docs = search_similar_docs(query, index, texts, k=3)
                rag_context = "\n\n".join(docs)
                logging.info(f"RAG context for highest temp query '{query}': {rag_context}")
            else:
                historical_response = analyze_historical_query(query)
                docs = search_similar_docs(query, index, texts, k=3)
                rag_context = "\n\n".join(docs)
                logging.info(f"RAG context for historical query '{query}': {rag_context}")

            context = f"Historical weather data for {city or 'unknown city'}: {historical_response}"
            response = generate_llm_response(query, context, city)
            st.chat_message("assistant").markdown(response, unsafe_allow_html=True)
            state.history.append({"role": "assistant", "content": response})
            st.stop()
        except Exception as e:
            logging.error(f"Historical query failed: {e}")
            response = f"⚠️ Unable to fetch historical data for {city.title() if city else 'this query'}. Please check Meteostat data availability or network connection."
            st.chat_message("assistant").markdown(response)
            state.history.append({"role": "assistant", "content": response})
            st.stop()

    # Handle weather and activity queries with RAG and LLM
    if intent in ("weather_query", "activity_query"):
        if not city and not state.last_city:
            response = f"🌍 Please tell me which city you'd like the weather for. Supported cities: {', '.join(c.title() for c in KNOWN_CITIES)}."
            state.pending_query = query
            st.chat_message("assistant").markdown(response)
            state.history.append({"role": "assistant", "content": response})
            st.stop()

        city = city or state.last_city
        # Fetch RAG context
        with st.spinner("☁️ Retrieving relevant information…"):
            try:
                docs = search_similar_docs(query, index, texts, k=3)
                rag_context = "\n\n".join(docs)
                logging.info(f"RAG context for query '{query}': {rag_context}")
            except Exception as e:
                logging.error(f"RAG search failed: {e}")
                rag_context = "No relevant documents found."

        # Handle direct Yes/No responses and highest temperature queries
        response_parts = []
        if intent == "weather_query" and "highest temperature" in query_lower:
            offset_days = 1 if "tomorrow" in query_lower else 0
            max_temp = get_max_temp_today(city, offset_days)
            if max_temp.startswith("⚠️"):
                response_parts.append(max_temp)
            else:
                direct_answer = f"**The highest temperature expected in {city.title()} {'tomorrow' if offset_days else 'today'} is {max_temp.split(',')[0]}.**"
                response_parts.append(direct_answer)
                response_parts.append(f"**{city.title()} {'Tomorrow’s' if offset_days else 'Today’s'} Forecast (OpenWeather API):** {max_temp}")
            # Generate additional LLM response with forecast data
            additional_response = generate_llm_response(query, rag_context, city, forecast_data=max_temp)
            response_parts.append(additional_response)
        elif intent == "weather_query" and "will it rain" in query_lower:
            offset_days = 1 if "tomorrow" in query_lower else 0
            will_rain = will_rain_on(city, offset_days)
            direct_answer = f"**{'Yes' if will_rain else 'No'}, it {'is likely to' if will_rain else 'is not likely to'} rain {'tomorrow' if offset_days else 'today'} in {city.title()}.**"
            response_parts.append(direct_answer)
            forecast = summary_on(city, offset_days)
            if forecast.startswith("⚠️"):
                response_parts.append(f"**Unable to fetch {'tomorrow’s' if offset_days else 'today’s'} forecast for {city.title()} (OpenWeather API).**")
            else:
                response_parts.append(f"**{city.title()} {'Tomorrow’s' if offset_days else 'Today’s'} Forecast (OpenWeather API):** {forecast}")
            # Generate additional LLM response with forecast data
            additional_response = generate_llm_response(query, rag_context, city, forecast_data=forecast)
            response_parts.append(additional_response)
        elif intent == "activity_query" and "weekend" in query_lower:
            today = datetime.utcnow().weekday()
            days_to_saturday = (5 - today) % 7
            days_to_sunday = (6 - today) % 7
            will_rain = will_rain_on(city, days_to_saturday) or will_rain_on(city, days_to_sunday)
            direct_answer = f"**{'Yes' if not will_rain else 'No'}, it is likely to be {'sunny' if not will_rain else 'rainy'} this weekend in {city.title()}.**"
            response_parts.append(direct_answer)
            forecast = get_weekend_forecast(city)
            if forecast.startswith("⚠️"):
                response_parts.append(f"**Unable to fetch weekend forecast for {city.title()} (OpenWeather API).**")
            else:
                response_parts.append(f"**{city.title()} Weekend Forecast (OpenWeather API):** {forecast}")
            # Generate additional LLM response with forecast data
            additional_response = generate_llm_response(query, rag_context, city, forecast_data=forecast)
            response_parts.append(additional_response)
        else:
            # Standard weather/activity query
            forecast_data = summary_on(city, 0) if intent == "weather_query" else None
            response = generate_llm_response(query, rag_context, city, forecast_data=forecast_data)
            response_parts.append(response)

        # Combine and display response
        response = "\n\n".join(response_parts)
        st.chat_message("assistant").markdown(response, unsafe_allow_html=True)
        state.history.append({"role": "assistant", "content": response})
        state.pending_query = None
        st.stop()

    elif intent == "general_query":
        # Fetch RAG context
        with st.spinner("☁️ Retrieving relevant information…"):
            try:
                docs = search_similar_docs(query, index, texts, k=3)
                rag_context = "\n\n".join(docs)
                logging.info(f"RAG context for query '{query}': {rag_context}")
            except Exception as e:
                logging.error(f"RAG search failed: {e}")
                rag_context = "No relevant documents found."

        # Generate LLM response with RAG context
        response = generate_llm_response(query, rag_context, state.last_city)
        st.chat_message("assistant").markdown(response, unsafe_allow_html=True)
        state.history.append({"role": "assistant", "content": response})
        state.pending_query = None
        st.stop()

    # Fallback for unrecognized intents
    response = f"🌐 I’m sorry, I didn’t understand that. Could you rephrase or specify a city? Supported cities: {', '.join(c.title() for c in KNOWN_CITIES)}."
    st.chat_message("assistant").markdown(response)
    state.history.append({"role": "assistant", "content": response})
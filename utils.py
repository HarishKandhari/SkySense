import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")

def load_sample_docs(folder_path="data/sample_docs"):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def get_direct_weather_answer(user_query: str, city: str) -> str:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()
        if "weather" not in res or "main" not in res:
            return f"⚠️ Could not fetch weather for {city.title()}."

        condition = res["weather"][0]["main"].lower()
        description = res["weather"][0]["description"].capitalize()
        temp = res["main"]["temp"]
        humidity = res["main"]["humidity"]
        wind = res["wind"]["speed"]

        q = user_query.lower()
        if "rain" in q:
            return f"🌧️ Yes, it might rain today in {city.title()}." if "rain" in condition else f"☀️ No, there's no rain expected today in {city.title()}."

        if any(word in q for word in ["go", "bike", "walk", "outside", "picnic", "movie", "ride"]):
            if "rain" in condition:
                return f"☔ It might rain in {city.title()}. Consider an umbrella or postponing."
            elif temp >= 35:
                return f"🥵 It's hot in {city.title()} ({temp}°C). Stay hydrated!"
            elif temp <= 10:
                return f"🥶 It's cold in {city.title()} ({temp}°C). Dress warmly!"
            else:
                return f"✅ The weather in {city.title()} looks fine for your plan."

        return f"🌤️ The weather in {city.title()} is {description} with {temp}°C, {humidity}% humidity and {wind} m/s wind."
    except Exception:
        return f"⚠️ Unable to analyze the weather in {city.title()} right now."

def is_activity_query(text):
    activity_phrases = [
        "go for a ride", "take my bike", "go to the park",
        "can I go out", "should I carry an umbrella", "go for a walk",
        "outdoor plan", "picnic", "outside today", "travel", "outing",
        "plan a trip", "go out", "ride my cycle", "is it safe to go",
        "movie", "go outside"
    ]
    return any(phrase in text.lower() for phrase in activity_phrases)
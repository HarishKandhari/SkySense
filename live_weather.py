# live_weather.py

import requests
import os
import time
from datetime import datetime, timedelta
from meteostat import Point, Daily
from visuals import get_coordinates
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ─── CITY COORDINATES ────────────────────────────────────────────────────────
CITY_COORDINATES = {
    "delhi":     (28.6139, 77.2090),
    "mumbai":    (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946),
    "kolkata":   (22.5726, 88.3639),
    "chennai":   (13.0827, 80.2707),
    "hyderabad": (17.3850, 78.4867)
}

# ─── CURRENT WEATHER ─────────────────────────────────────────────────────────
def get_current_weather(city: str):
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    res = requests.get(url).json()
    if "weather" not in res or "main" not in res:
        return None, "⚠️ Weather not available."

    icon = res["weather"][0]["icon"]
    icon_url = f"http://openweathermap.org/img/wn/{icon}@2x.png"
    desc = res["weather"][0]["description"].capitalize()
    temp = res["main"]["temp"]
    humidity = res["main"]["humidity"]
    wind = res["wind"]["speed"]

    preview = f"**{desc}**, {temp}°C, 💧 {humidity}% humidity, 💨 {wind} m/s"
    return icon_url, preview

# ─── 5‑DAY FORECAST ──────────────────────────────────────────────────────────
def get_forecast(city: str) -> str:
    url = (
        f"http://api.openweathermap.org/data/2.5/forecast"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    res = requests.get(url).json()
    if "list" not in res:
        return "⚠️ Forecast data not available."

    daily, seen = [], set()
    for entry in res["list"]:
        date = entry["dt_txt"][:10]
        if date not in seen:
            seen.add(date)
            desc = entry["weather"][0]["description"].capitalize()
            temp = entry["main"]["temp"]
            daily.append((date, desc, temp))
        if len(daily) >= 5:
            break

    out = "### 📅 5‑Day Forecast\n"
    for d, desc, temp in daily:
        out += f"- {d}: **{desc}**, {temp}°C\n"
    return out

# ─── MAX TEMP LAST YEAR ─────────────────────────────────────────────────────
def get_max_temp_last_year(city: str) -> str:
    coords = get_coordinates(city)
    if not coords:
        return f"⚠️ Coordinates for {city.title()} not found."
    end = datetime.now()
    start = end - timedelta(days=365)
    df = Daily(Point(coords[0], coords[1]), start, end).fetch()
    if 'tmax' in df.columns and not df['tmax'].empty:
        max_temp = df['tmax'].max()
        date = df['tmax'].idxmax().strftime('%Y-%m-%d')
        return f"🌡️ The highest temperature in {city.title()} in the past year was {max_temp:.1f}°C on {date}."
    return f"⚠️ Temperature data not available for {city.title()}."


# ─── LAST RAIN EVENT ─────────────────────────────────────────────────────────
def get_last_rain_event(city: str) -> str:
    coords = CITY_COORDINATES.get(city.lower())
    if not coords:
        return f"⚠️ Coordinates for {city.title()} not found."
    lat, lon = coords

    now = int(time.time())
    rain_timestamps = []

    # check ≈1 h ago and ≈24 h ago
    for delta in (3600, 3600 * 24):
        dt = now - delta
        url = (
            f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
            f"?lat={lat}&lon={lon}&dt={dt}"
            f"&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        res = requests.get(url).json()
        for hour in res.get("hourly", []):
            if hour.get("rain") or hour["weather"][0]["main"].lower() == "rain":
                rain_timestamps.append(hour["dt"])

    if not rain_timestamps:
        return f"🌧️ No rain detected in the last 48 hours for {city.title()}."

    last_ts = max(rain_timestamps)
    dt_obj = datetime.utcfromtimestamp(last_ts)
    timestr = dt_obj.strftime("%Y-%m-%d %H:%M UTC")
    return f"🌧️ Last rain in {city.title()} was on {timestr}."

# ─── YESTERDAY’S TEMPERATURE ───────────────────────────────────────────────
def get_yesterday_temperature(city: str) -> str:
    coords = CITY_COORDINATES.get(city.lower())
    if not coords:
        return f"⚠️ Coordinates for {city.title()} not found."
    lat, lon = coords
    ts = int(time.time()) - 86400  # 24h ago
    url = (
        f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
        f"?lat={lat}&lon={lon}&dt={ts}"
        f"&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    res = requests.get(url).json()
    hours = res.get("hourly", [])
    if not hours:
        return f"⚠️ No historical data available for {city.title()}."

    # pick the hour closest to current UTC hour
    now_h = datetime.utcfromtimestamp(time.time()).hour
    entry = min(hours, key=lambda h: abs(datetime.utcfromtimestamp(h["dt"]).hour - now_h))
    temp = entry["temp"]
    dt_str = datetime.utcfromtimestamp(entry["dt"]).strftime("%Y-%m-%d %H:%M UTC")
    return f"🌡️ Yesterday in {city.title()} at {dt_str} it was **{temp:.1f}°C**."

# ─── 7‑DAY FORECAST ───────────────────────────────────────────────────────────
def get_weekly_forecast(city: str) -> str:
    coords = CITY_COORDINATES.get(city.lower())
    if not coords:
        return f"⚠️ Coordinates for {city.title()} not found."
    lat, lon = coords

    url = (
        f"https://api.openweathermap.org/data/2.5/onecall"
        f"?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts"
        f"&units=metric&appid={OPENWEATHER_API_KEY}"
    )
    res = requests.get(url).json()
    if "daily" not in res:
        return "⚠️ Weekly forecast not available."

    out = "### 📅 7‑Day Forecast\n"
    for day in res["daily"][1:8]:
        date = datetime.utcfromtimestamp(day["dt"]).date().isoformat()
        desc = day["weather"][0]["description"].capitalize()
        tmax = day["temp"]["max"]
        tmin = day["temp"]["min"]
        out += f"- {date}: **{desc}**, High {tmax}°C, Low {tmin}°C\n"
    return out

# ─── AIR QUALITY ────────────────────────────────────────────────────────────
def get_air_quality(city: str) -> str:
    coords = CITY_COORDINATES.get(city.lower())
    if not coords:
        return f"⚠️ Coordinates for {city.title()} not found."
    lat, lon = coords

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    )
    lst = requests.get(url).json().get("list", [])
    if not lst:
        return "⚠️ Air quality data not available."

    aqi = lst[0]["main"]["aqi"]
    aqi_map = {1:"Good",2:"Fair",3:"Moderate",4:"Poor",5:"Very Poor"}
    return f"🌐 Air Quality Index for {city.title()} is {aqi_map.get(aqi,'Unknown')} (AQI {aqi})."

# ─── UV INDEX ────────────────────────────────────────────────────────────────
def get_uv_index(city: str) -> str:
    coords = CITY_COORDINATES.get(city.lower())
    if not coords:
        return f"⚠️ Coordinates for {city.title()} not found."
    lat, lon = coords

    url = (
        f"https://api.openweathermap.org/data/2.5/onecall"
        f"?lat={lat}&lon={lon}&exclude=minutely,hourly,daily,alerts"
        f"&appid={OPENWEATHER_API_KEY}"
    )
    current = requests.get(url).json().get("current", {})
    uvi = current.get("uvi")
    if uvi is None:
        return "⚠️ UV index data not available."

    if uvi <= 2:
        cat = "Low"
    elif uvi <= 5:
        cat = "Moderate"
    elif uvi <= 7:
        cat = "High"
    elif uvi <= 10:
        cat = "Very High"
    else:
        cat = "Extreme"

    return f"🌞 UV Index in {city.title()} is {uvi} ({cat})."

# ─── YESTERDAY'S WINDSPEED ────────────────────────────────────────────────────
def get_yesterday_windspeed(city: str) -> str:
    coords = CITY_COORDINATES.get(city.lower())
    if not coords:
        return f"⚠️ Coordinates for {city.title()} not found."
    lat, lon = coords

    # fetch last 48h of hourly data
    url = (
        f"https://api.openweathermap.org/data/2.5/onecall"
        f"?lat={lat}&lon={lon}&exclude=current,minutely,daily,alerts"
        f"&units=metric&appid={OPENWEATHER_API_KEY}"
    )
    res = requests.get(url).json()
    hourly = res.get("hourly", [])
    if not hourly:
        return f"⚠️ No hourly data available for {city.title()}."

    # target ~24h ago
    target = datetime.utcnow() - timedelta(days=1)
    closest = min(hourly, key=lambda e: abs(datetime.utcfromtimestamp(e["dt"]) - target))
    ws = closest.get("wind_speed")
    dt = datetime.utcfromtimestamp(closest["dt"])
    if ws is None:
        return f"⚠️ Wind speed data not available for {city.title()} yesterday."
    return f"💨 Yesterday’s wind speed in {city.title()} was {ws} m/s at {dt.strftime('%Y-%m-%d %H:%M UTC')}."
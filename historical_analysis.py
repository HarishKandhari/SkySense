import re
from datetime import datetime, timedelta
import pandas as pd
from meteostat import Point, Daily
import plotly.graph_objects as go
from io import BytesIO
import base64
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Supported cities
CITY_COORDINATES = {
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946),
    "kolkata": (22.5726, 88.3639),
    "chennai": (13.0827, 80.2707),
    "hyderabad": (17.3850, 78.4867)
}

HISTORICAL_PATTERNS = [
    r"hotter than last year",
    r"cooler than last year",
    r"has.*rainfall.*(decreased|increased)",
    r"has.*humidity.*changed",
    r"(compare|comparison).*(last year|5 years ago)",
    r"how.*changed.*over.*years",
    r"(temperature|humidity|rainfall).*past.*(year|5 years)"
]

def is_historical_query(query: str) -> bool:
    query_lower = query.lower()
    if any(re.search(p, query_lower) for p in HISTORICAL_PATTERNS):
        return True
    try:
        classification = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a classifier that answers only 'yes' or 'no'. The user will ask a question."},
                {"role": "user", "content": f"Does the following ask about historical weather trends like past temperature, rainfall, or humidity?\n\n{query}"}
            ],
            temperature=0
        ).choices[0].message.content.strip().lower()
        return "yes" in classification
    except:
        return False

def extract_city(query: str) -> str:
    for city in CITY_COORDINATES:
        if city in query.lower():
            return city
    return None

def fetch_weather_data(city: str, years: int = 5) -> pd.DataFrame:
    lat, lon = CITY_COORDINATES[city]
    location = Point(lat, lon)
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    df = Daily(location, start, end).fetch()
    return df

def create_combination_chart(df: pd.DataFrame, city: str) -> str:
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['time']).dt.date

    fig = go.Figure()
    if 'tavg' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['tavg'], mode='lines', name='Avg Temp (°C)'))
    if 'prcp' in df.columns:
        fig.add_trace(go.Bar(x=df['date'], y=df['prcp'], name='Rainfall (mm)', yaxis='y2'))
    if 'rhum' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['rhum'], mode='lines', name='Humidity (%)', yaxis='y3'))

    fig.update_layout(
        title=f"📈 Historical Weather Trends – {city.title()}",
        xaxis_title="Date",
        yaxis=dict(title="Avg Temp (°C)", side="left"),
        yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right"),
        yaxis3=dict(title="Humidity (%)", overlaying="y", side="left", anchor="free", position=0.05),
        legend=dict(x=0.01, y=1.0),
        height=500
    )

    buf = BytesIO()
    fig.write_image(buf, format='png')
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{encoded}" width="100%">'

def analyze_historical_query(query: str) -> str:
    city = extract_city(query)
    if not city:
        return "⚠️ I couldn't detect the city in your question. Please mention a supported city."

    try:
        df = fetch_weather_data(city)
    except Exception as e:
        return f"❌ Unable to retrieve data: {e}"

    q = query.lower()
    lines = []

    # Temperature analysis
    if "temperature" in q or "hotter" in q or "cooler" in q:
        this_year = df[df.index >= datetime.now() - timedelta(days=30)]['tavg'].mean()
        last_year = df[(df.index >= datetime.now() - timedelta(days=395)) & (df.index <= datetime.now() - timedelta(days=365))]['tavg'].mean()
        diff = this_year - last_year
        trend = "hotter" if diff > 0 else "cooler"
        lines.append(f"🌡️ This summer in {city.title()} is about **{abs(diff):.1f}°C {trend}** than the same time last year.")

    # Rainfall analysis
    if "rain" in q or "monsoon" in q:
        recent_rain = df[df.index >= datetime.now() - timedelta(days=365)]['prcp'].sum()
        past_rain = df[(df.index >= datetime.now() - timedelta(days=730)) & (df.index <= datetime.now() - timedelta(days=365))]['prcp'].sum()
        trend = "increased" if recent_rain > past_rain else "decreased"
        lines.append(f"🌧️ Monsoon rainfall in {city.title()} has **{trend}** compared to the previous year.")

    # Humidity analysis
    if "humidity" in q or "logistics" in q:
        if 'rhum' in df.columns:
            recent_hum = df[df.index >= datetime.now() - timedelta(days=365)]['rhum'].mean()
            past_hum = df[(df.index >= datetime.now() - timedelta(days=730)) & (df.index <= datetime.now() - timedelta(days=365))]['rhum'].mean()
            diff = recent_hum - past_hum
            trend = "increased" if diff > 0 else "decreased"
            lines.append(f"💧 Humidity in {city.title()} has **{trend} by {abs(diff):.1f}%** over the past year.")
        else:
            lines.append("⚠️ Humidity data not available for comparison.")

    if not lines:
        lines.append("📊 I understood this as a historical question, but couldn’t find a clear indicator. Try asking about temperature, rainfall, or humidity trends.")

    chart = create_combination_chart(df.tail(365), city)
    return "\n\n".join(lines) + "\n\n" + chart

# app.py
import os, sys, pickle, subprocess

VEC_PATH = "data/vector_index.pkl"
if not os.path.exists(VEC_PATH):
    print(f"⚠️ No FAISS index found at {VEC_PATH}, building now…")
    subprocess.run([sys.executable, "build_index.py"], check=True)

try:
    with open(VEC_PATH, "rb") as f:
        index, texts = pickle.load(f)
except Exception as e:
    print(f"❌ Failed to load FAISS index: {e}")
    sys.exit(1)

from retriever import get_embedding, search_similar_docs
import streamlit as st

# ─── 0. LOAD ENV & KEYS ──────────────────────────────────────────────────────
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(), override=True)

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
MAPBOX_API_KEY      = os.getenv("MAPBOX_API_KEY")
SERPAPI_KEY         = os.getenv("SERPAPI_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("⚠️ OPENAI_API_KEY not found in .env")
if not MAPBOX_API_KEY:
    raise RuntimeError("⚠️ MAPBOX_API_KEY not found in .env")

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import re, json, pickle, requests, difflib, streamlit as st
from datetime import datetime, timedelta
from openai import OpenAI
from meteostat import Point, Daily
import math

from utils import get_direct_weather_answer
from historical_analysis import is_historical_query, analyze_historical_query
from live_weather import (
    get_current_weather,
    get_air_quality,
    get_uv_index,
    get_forecast,
    get_last_rain_event,
    get_weekly_forecast,
    get_yesterday_windspeed
)
from news_sentiment import get_weather_news_and_sentiment
from visuals import plot_weather_metrics, plot_monthly_trends, get_coordinates
from compare import compare_weather
from retriever import search_similar_docs
from rag_engine import get_rag_response

# ─── FORECAST HELPERS ────────────────────────────────────────────────────────
def fetch_5day_json(city: str) -> dict:
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    return requests.get(url).json()

def get_5day_forecast(city: str) -> str:
    data = fetch_5day_json(city)
    if "list" not in data: return "⚠️ Forecast data not available."
    daily, seen = [], set()
    for e in data["list"]:
        d = e["dt_txt"][:10]
        if d not in seen:
            seen.add(d)
            desc = e["weather"][0]["description"].capitalize()
            temp = e["main"]["temp"]
            daily.append((d, desc, temp))
        if len(daily) >= 5:
            break
    out = "### 📅 5‑Day Forecast\n"
    for d, desc, temp in daily:
        out += f"- {d}: **{desc}**, {temp}°C\n"
    return out

def will_rain_on(city: str, offset_days: int) -> bool:
    data = fetch_5day_json(city)
    if "list" not in data: return False
    target = (datetime.utcnow() + timedelta(days=offset_days)).date().isoformat()
    for e in data["list"]:
        if e["dt_txt"].startswith(target):
            main = e["weather"][0]["main"].lower()
            pop  = e.get("pop", 0)
            if pop > 0.2 or "rain" in main:
                return True
    return False

def summary_on(city: str, offset_days: int) -> str:
    data = fetch_5day_json(city)
    if "list" not in data: return "⚠️ Forecast data not available."
    target = (datetime.utcnow() + timedelta(days=offset_days)).date().isoformat()
    cands = [e for e in data["list"] if e["dt_txt"].startswith(target)]
    if not cands:
        return f"⚠️ No forecast for {target}."
    entry = min(cands, key=lambda e: abs(int(e["dt_txt"][11:13]) - 12))
    desc = entry["weather"][0]["description"].capitalize()
    temp = entry["main"]["temp"]
    return f"{desc}, {temp}°C"

# ─── COMPARISON SUMMARY ──────────────────────────────────────────────────────
def compute_comparison_summary(c1: str, c2: str) -> str:
    end   = datetime.now()
    start = end - timedelta(days=365)

    lat1, lon1 = get_coordinates(c1)
    lat2, lon2 = get_coordinates(c2)
    df1 = Daily(Point(lat1, lon1), start, end).fetch()
    df2 = Daily(Point(lat2, lon2), start, end).fetch()

    t1 = df1["tavg"].mean() if "tavg" in df1.columns else math.nan
    t2 = df2["tavg"].mean() if "tavg" in df2.columns else math.nan
    p1 = df1["prcp"].sum()  if "prcp" in df1.columns else math.nan
    p2 = df2["prcp"].sum()  if "prcp" in df2.columns else math.nan

    temp_text = f"{t1:.1f}°C vs {t2:.1f}°C" if not math.isnan(t1) and not math.isnan(t2) else "coming soon 😞"
    prcp_text = f"{p1:.0f} mm vs {p2:.0f} mm" if not math.isnan(p1) and not math.isnan(p2) else "coming soon 😞"

    return (
        f"Over the past year, **{c1}** averaged **{temp_text}** in average temperature.  \n"
        f"Annual precipitation was **{prcp_text}**.  \n"
        f"Humidity comparison coming soon. 😞"
    )

# ─── INTENT PARSER ──────────────────────────────────────────────────────────
def parse_user_message(message: str, last_city: str=None) -> dict:
    system = (
        "You are a router for a weather assistant. Output only JSON with keys:\n"
        "intent ∈ [greeting,weather_query,activity_query,historical_query,compare_query,rag_fallback],\n"
        "city: str or null,\n"
        "cities: list of two city names.\n"
    )
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":message}],
        temperature=0
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"intent": None, "city": None, "cities": []}

# ─── SETUP ───────────────────────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)
st.set_page_config(page_title="SkySense", layout="wide")
st.title("🌤️ SkySense – RAG‑Based Weather Chatbot")

with open("data/vector_index.pkl", "rb") as f:
    index, texts = pickle.load(f)

state = st.session_state
state.setdefault("history", [])
state.setdefault("last_city", None)
state.setdefault("city_set_by_chat", False)
state.setdefault("pending_query", None)
state.setdefault("view_switch", "📅 Forecast")
state.setdefault("mode", "💬 Chat Mode")

KNOWN_CITIES = ["Delhi","Mumbai","Bangalore","Kolkata","Chennai","Hyderabad"]

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🌍 Weather Options")
    state.mode = st.radio("Mode", ["💬 Chat Mode","📊 Tool Mode"],
                         index=0 if state.mode=="💬 Chat Mode" else 1)
    choice = st.selectbox("City", KNOWN_CITIES,
                          index=KNOWN_CITIES.index(state.last_city.title()) if state.last_city else 0)
    state.last_city = choice.lower()
    try:
        ico, prev = get_current_weather(state.last_city)
        st.image(ico, width=60); st.markdown(prev)
    except:
        st.markdown("⚠️ Weather not available.")
    for lbl in ["📅 Forecast","📈 Monthly Trends","📰 News Summary","📊 Charts","🏭 Air Quality","🌞 UV Index","🌐 Compare Cities"]:
        if st.button(lbl):
            state.view_switch = lbl

# ─── SHOW HISTORY ─────────────────────────────────────────────────────────────
for msg in state.history:
    st.chat_message(msg["role"]).markdown(msg["content"], unsafe_allow_html=True)

# ─── TOOL MODE ───────────────────────────────────────────────────────────────
if state.mode == "📊 Tool Mode":
    st.subheader(f"{state.view_switch} – {state.last_city.title()}")

    if state.view_switch == "📅 Forecast":
        st.markdown(get_5day_forecast(state.last_city), unsafe_allow_html=True)
        lines = [
            f"- **{(datetime.utcnow() + timedelta(days=i)).date()}:** {summary_on(state.last_city, i)}"
            for i in range(1, 6)
        ]
        st.markdown("### 📖 Forecast Summary\n" + "\n".join(lines))

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
        sel = st.multiselect("Choose 2+ cities", KNOWN_CITIES, default=[state.last_city.title()])
        if len(sel) >= 2:
            compare_weather(sel)
q=" "
ql=" "
# ─── CHAT INPUT & LOGIC ──────────────────────────────────────────────────────
if user_input := st.chat_input("Ask me anything…"):
    st.chat_message("user").markdown(user_input)
    state.history.append({"role":"user","content":user_input})
    q = user_input.strip()
    ql = user_input.strip().lower()

    # ——— Map request handler ———
    if "map" in ql:
        m = re.search(r"map of\s+([\w\s]+)", ql)
        place = m.group(1).strip() if m else None
        if not place and not state.last_city:
            prompt = "🌍 Please tell me which place you'd like to see the map of."
            st.chat_message("assistant").markdown(prompt)
            state.history.append({"role":"assistant","content":prompt})
            st.stop()
        loc = (place or state.last_city).title()

        geo_resp = requests.get(
            f"https://api.mapbox.com/geocoding/v5/mapbox.places/{requests.utils.quote(loc)}.json"
            f"?access_token={MAPBOX_API_KEY}&limit=1"
        ).json()
        features = geo_resp.get("features", [])
        if not features:
            msg = f"⚠️ Could not find '{loc}'."
            st.chat_message("assistant").markdown(msg)
            state.history.append({"role":"assistant","content":msg})
            st.stop()

        lon, lat = features[0]["center"]
        map_url = (
            f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/"
            f"static/{lon},{lat},10,0/600x400"
            f"?access_token={MAPBOX_API_KEY}"
        )
        st.chat_message("assistant").image(map_url, caption=f"Map of {loc}")
        state.history.append({"role":"assistant","content":f"Displayed map of {loc}."})
        st.stop()

    # ─── Re‑extract any city mentioned in this query ─────────────────────────
    parsed = parse_user_message(q, last_city=state.last_city)
    explicit = parsed.get("city")
    if not explicit:
        for c in KNOWN_CITIES:
            if c.lower() in ql:
                explicit = c.lower()
                break
    explicit = explicit or state.last_city

    # ─── Flexible “last rain” catch‑all ─────────────────────────────────────
    if explicit and re.search(r"(last.*rain|did.*rain)", ql):
        ans = get_last_rain_event(explicit)
        st.chat_message("assistant").markdown(ans)
        state.history.append({"role":"assistant","content":ans})
        st.stop()

    # ─── Forecast‑for‑today branch (via OneCall daily[0]) ────────────────────────
    if explicit and "forecast" in ql and "today" in ql:
        # onecall daily endpoint
        lat, lon = CITY_COORDINATES.get(explicit.lower(), (None, None))
        if lat is None:
            ans = f"⚠️ Coordinates for {explicit.title()} not found."
        else:
            oc = requests.get(
                f"https://api.openweathermap.org/data/2.5/onecall"
                f"?lat={lat}&lon={lon}"
               f"&exclude=minutely,hourly,alerts"
                f"&units=metric&appid={OPENWEATHER_API_KEY}"
            ).json()
            today = oc.get("daily", [])[0]
            desc = today["weather"][0]["description"].capitalize()
            tmax = today["temp"]["max"]
            tmin = today["temp"]["min"]
            ans = (
                f"📅 Today in {explicit.title()}: {desc}.  \n"
                f"🔺 High: {tmax:.1f}°C · 🔻 Low: {tmin:.1f}°C"
            )
        st.chat_message("assistant").markdown(ans)
        state.history.append({"role":"assistant","content":ans})
        st.stop()


    # ─── Highest temp last year ───────────────────────────────────────────────
    if explicit \
       and "highest temperature" in ql \
       and ("last year" in ql or "past year" in ql):
        ans = get_max_temp_last_year(explicit)
        st.chat_message("assistant").markdown(ans)
        state.history.append({"role":"assistant","content":ans})
        st.stop()
    

    # 1️⃣ “What about tomorrow?”
    ql_clean = re.sub(r"[^\w\s]", "", ql).strip()
    if state.city_set_by_chat and ql_clean in ("what about tomorrow", "how about tomorrow", "tomorrow"):
        will = will_rain_on(state.last_city, 1)
        ans = (
            f"🌧️ Yes, rain expected tomorrow in {state.last_city.title()}."
            if will
            else f"☀️ No, there's no rain expected tomorrow in {state.last_city.title()}."
        )
        st.chat_message("assistant").markdown(ans)
        state.history.append({"role":"assistant","content":ans})
        st.stop()

    # 2️⃣ Last rain event
    if state.city_set_by_chat and "last time" in ql and "rain" in ql:
        ans = get_last_rain_event(state.last_city)
        st.chat_message("assistant").markdown(ans)
        state.history.append({"role":"assistant","content":ans})
        st.stop()

    # 3️⃣ Next‑week forecast
    if state.city_set_by_chat and ("next week" in ql or "next weekend" in ql):
        wf = get_weekly_forecast(state.last_city)
        st.chat_message("assistant").markdown(wf, unsafe_allow_html=True)
        state.history.append({"role":"assistant","content":wf})
        st.stop()

    # 4️⃣ Monthly trends (Chat)
    if state.city_set_by_chat and ("monthly trend" in ql or "monthly weather trend" in ql):
        plot_monthly_trends(state.last_city)
        state.history.append({"role":"assistant","content":"Displayed monthly trends."})
        st.stop()

    # 5️⃣ Yesterday's windspeed
    if state.city_set_by_chat and "yesterday" in ql and "windspeed" in ql:
        ans = get_yesterday_windspeed(state.last_city)
        st.chat_message("assistant").markdown(ans)
        state.history.append({"role":"assistant","content":ans})
        st.stop()

    # ─── Yesterday’s temperature ───────────────────────────────────────────────
    if state.city_set_by_chat and "yesterday" in ql and "temperature" in ql:
        ans = get_yesterday_temperature(state.last_city)
        st.chat_message("assistant").markdown(ans)
        state.history.append({"role":"assistant","content":ans})
        st.stop()

    # 6️⃣ Current windspeed
    if state.city_set_by_chat and "windspeed" in ql:
        _, preview = get_current_weather(state.last_city)
        wind_part = preview.split("💨")[-1].strip()
        ans = f"💨 Current wind speed in {state.last_city.title()} is {wind_part}"
        st.chat_message("assistant").markdown(ans)
        state.history.append({"role":"assistant","content":ans})
        st.stop()

    # 7️⃣ “Did you mean …?” confirmation
    if state.get("suggested_city"):
        if ql == "yes":
            city = state.pop("suggested_city")
            state.last_city = city
            state.city_set_by_chat = True
            ico, prev = get_current_weather(city)
            out = f"**{city.title()} – Live Weather**\n\n{prev}"
            st.chat_message("assistant").markdown(out, unsafe_allow_html=True)
            state.history.append({"role":"assistant","content":out})
        elif ql == "no":
            state.pop("suggested_city")
            msg = "Okay, please tell me the city again."
            st.chat_message("assistant").markdown(msg)
            state.history.append({"role":"assistant","content":msg})
        st.stop()

    # 8️⃣ Typo suggestion
    parsed = parse_user_message(q, last_city=state.last_city)
    intent = parsed.get("intent")
    explicit = parsed.get("city")

    # 9️⃣ New city provided (explicit city in user message)
    if explicit:
        state.last_city = explicit
        state.city_set_by_chat = True

        # show live‑weather header
        ico, prev = get_current_weather(explicit)
        live = f"**{explicit.title()} – Live Weather**\n\n{prev}"
        st.chat_message("assistant").markdown(live, unsafe_allow_html=True)
        state.history.append({"role":"assistant","content":live})

        # consume pending query if any
        if state.pending_query:
            ans_p = get_direct_weather_answer(state.pending_query, explicit)
            st.chat_message("assistant").markdown(ans_p)
            state.history.append({"role":"assistant","content":ans_p})
            state.pending_query = None
            st.stop()

        # if weather/activity, handle tomorrow branch
        if intent in ("weather_query","activity_query"):
            if "tomorrow" in ql:
                summ = summary_on(explicit, 1)
                ans = f"Tomorrow in {explicit.title()}: {summ}"
            else:
                ans = get_direct_weather_answer(q, explicit)
            st.chat_message("assistant").markdown(ans)
            state.history.append({"role":"assistant","content":ans})
        st.stop()

    # 10️⃣ Compare cities
    if intent == "compare_query" and len(parsed.get("cities", [])) == 2:
        c1, c2 = [c.title() for c in parsed["cities"]]
        compare_weather([c1, c2])
        summ = compute_comparison_summary(c1, c2)
        st.chat_message("assistant").markdown(f"**Summary:** {summ}")
        state.history.append({"role":"assistant","content":summ})
        st.stop()

    # 11️⃣ Historical trends
    if intent == "historical_query":
        out = analyze_historical_query(q)
        st.chat_message("assistant").markdown(out, unsafe_allow_html=True)
        state.history.append({"role":"assistant","content":out})
        st.stop()

    # 12️⃣ Weather / Activity fallback
    if intent in ("weather_query","activity_query"):
        state.pending_query = q
        ask = "🌍 Please tell me which city you'd like the weather for."
        st.chat_message("assistant").markdown(ask)
        state.history.append({"role":"assistant","content":ask})
        st.stop()

    # 13️⃣ Greeting
    if intent == "greeting":
        with st.spinner("☁️ SkySense thinking…"):
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role":"system","content":"You are SkySense, a friendly weather assistant."},
                    {"role":"user","content":q},
                    {"role":"assistant","content":"Greet warmly and ask for the user's city."}
                ],
                temperature=0.7
            ).choices[0].message.content.strip()
        st.chat_message("assistant").markdown(resp)
        state.history.append({"role":"assistant","content":resp})
        st.stop()

     
    # ─── 14️⃣ RAG fallback ─────────────────────────────────────────────────────
with st.spinner("☁️ SkySense thinking…"):
    docs = search_similar_docs(q, index, texts, k=3)
    #Generate the RAG response
    rag = get_rag_response(q, docs)

    #Prepend live weather if needed
    if state.city_set_by_chat:
        ico, prev = get_current_weather(state.last_city)
        rag = (
            f"<img src='{ico}' width='50'> "
            f"**{state.last_city.title()} – Live Weather**\n\n{prev}\n\n"
        ) + rag

# 6. Render and record it
st.chat_message("assistant").markdown(rag, unsafe_allow_html=True)
state.history.append({"role":"assistant","content":rag})
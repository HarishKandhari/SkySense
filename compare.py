import streamlit as st
from meteostat import Point, Daily
import plotly.graph_objects as go
from datetime import datetime, timedelta

def compare_weather(cities):
    st.markdown("### 🌐 Weather Comparison")

    end = datetime.now()
    start = end - timedelta(days=365)

    city_data = {}

    for city in cities:
        coords = get_coordinates(city)
        if coords:
            location = Point(coords[0], coords[1])
            data = Daily(location, start, end)
            data = data.fetch()
            if not data.empty:
                city_data[city.title()] = data

    if len(city_data) < 2:
        st.warning("⚠️ Could not fetch data for multiple cities.")
        return

    plot_comparison_charts(city_data)

def plot_comparison_charts(city_data):
    metrics = {
        "Average Temperature (°C)": "tavg",
        "Max Temperature (°C)": "tmax",
        "Min Temperature (°C)": "tmin",
        "Humidity (%)": "rhum",
        "Wind Speed (km/h)": "wspd",
        "Precipitation (mm)": "prcp"
    }

    for label, key in metrics.items():
        fig = go.Figure()
        for city, df in city_data.items():
            if key in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[key],
                    mode="lines",
                    name=city
                ))
        fig.update_layout(
            title=f"📊 {label} – Last 1 Year",
            xaxis_title="Date",
            yaxis_title=label,
            legend_title="City",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📈 Data Source: Meteostat")

def get_coordinates(city):
    known_coords = {
        "Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Bangalore": (12.9716, 77.5946),
        "Kolkata": (22.5726, 88.3639),
        "Chennai": (13.0827, 80.2707),
        "Hyderabad": (17.3850, 78.4867)
    }
    return known_coords.get(city.title())
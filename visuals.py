# visuals.py

from datetime import datetime, timedelta
from meteostat import Point, Daily
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def plot_weather_metrics(city):
    st.markdown(f"### 📊 Weather Charts – {city.title()}")

    coords = get_coordinates(city)
    if not coords:
        st.warning("⚠️ Coordinates not found for this city.")
        return

    location = Point(coords[0], coords[1])
    end = datetime.now()
    start = end - timedelta(days=365)

    data = Daily(location, start, end).fetch()
    if data.empty:
        st.warning("⚠️ No weather data available.")
        return

    metrics = {
        "Average Temperature (°C)": ("tavg", "orange"),
        "Max Temperature (°C)": ("tmax", "red"),
        "Min Temperature (°C)": ("tmin", "blue"),
        "Humidity (%)": ("rhum", "green"),
        "Precipitation (mm)": ("prcp", "purple"),
        "Wind Speed (km/h)": ("wspd", "gray"),
        "Dew Point (°C)": ("dwpt", "cyan"),
        "Snowfall (cm)": ("snow", "lightblue"),
    }

    shown = 0
    for title, (key, color) in metrics.items():
        if key not in data.columns or data[key].isna().all():
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, y=data[key], mode="lines",
            line=dict(color=color), name=key
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=title,
            margin=dict(l=30, r=30, t=40, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📊 Source: Meteostat")
        shown += 1

    if shown == 0:
        st.warning("⚠️ No valid charts could be generated due to missing data.")

def plot_monthly_trends(city):
    st.markdown(f"### 📈 Monthly Weather Trends – {city.title()}")

    coords = get_coordinates(city)
    if not coords:
        st.error(f"⚠️ Coordinates for {city.title()} not found.")
        return

    # Fetch daily data for the past year
    end = datetime.now()
    start = end - timedelta(days=365)
    df = Daily(Point(coords[0], coords[1]), start, end).fetch()

    if df.empty:
        st.warning("⚠️ No weather data available for monthly trends.")
        return

    # Prepare month column
    df = df.reset_index()
    df['month'] = df['time'].dt.to_period('M')

    # Aggregate by month
    monthly = df.groupby('month').agg({
        'tavg': 'mean',
        'prcp': 'sum'
    }).reset_index()
    monthly['month_str'] = monthly['month'].dt.strftime('%Y-%m')

    # Plot Average Temperature per Month
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=monthly['month_str'], y=monthly['tavg'], name='Avg Temp (°C)'
    ))
    fig1.update_layout(
        title=f"📊 Monthly Average Temperature – {city.title()}",
        xaxis_title="Month",
        yaxis_title="Avg Temp (°C)",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Plot Total Precipitation per Month
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=monthly['month_str'], y=monthly['prcp'], name='Total Precipitation (mm)'
    ))
    fig2.update_layout(
        title=f"📊 Monthly Total Precipitation – {city.title()}",
        xaxis_title="Month",
        yaxis_title="Precipitation (mm)",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

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
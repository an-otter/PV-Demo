from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.graph_objects as go
from datetime import date, timedelta

BASE_DIR = Path(__file__).resolve().parent

LAT, LON = 52.22, 13.20
WETTER_VARS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
               "temperature_2m", "relative_humidity_2m", "cloud_cover"]

# Beispiel Verbrauch
KWH_WASCHMASCHINE = 0.8   
KWH_GESCHIRR      = 1.2    
KWH_EAUTO_KM      = 0.16   
KWH_LAPTOP_H      = 0.05   
KWH_HAUSHALT_TAG  = 8.0     

st.set_page_config(page_title="PV-Ertragsprognose", page_icon="☀️", layout="wide")


@st.cache_resource
def lade_modell():
    paket = joblib.load(BASE_DIR / "pv_modell_deploy.pkl")
    return paket["modell"], paket["features"]


modell, features = lade_modell()


@st.cache_data(ttl=3600)
def hole_wetter(tag: date) -> pd.DataFrame:
    params = {"latitude": LAT, "longitude": LON, "hourly": ",".join(WETTER_VARS),
              "timezone": "UTC", "start_date": tag.isoformat(), "end_date": tag.isoformat()}
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    if df[WETTER_VARS].isna().any().any():
        raise ValueError("Unvollständige Wetterdaten")
    return df


def baue_features(wetter: pd.DataFrame) -> pd.DataFrame:
    df = wetter.copy()
    stunde, tag = df.index.hour, df.index.dayofyear
    df["stunde_sin"] = np.sin(2 * np.pi * stunde / 24)
    df["stunde_cos"] = np.cos(2 * np.pi * stunde / 24)
    df["tag_sin"]    = np.sin(2 * np.pi * tag / 365)
    df["tag_cos"]    = np.cos(2 * np.pi * tag / 365)
    return df[features]


def de(n: float) -> str:
    """Ganzzahl"""
    return f"{n:,.0f}".replace(",", ".")


st.title("Demo-PV-Vorhersage")
st.caption("Stündliche Ertragsprognose · Modell: Gradient Boosting, Variante B (nur reale Messdaten)")

heute = date.today()
steuerung, _ = st.columns([1, 3])
with steuerung:
    gewaehlter_tag = st.date_input("Prognosetag", value=heute + timedelta(days=1),
                                   min_value=heute - timedelta(days=70),
                                   max_value=heute + timedelta(days=14))
    demo_modus = st.checkbox("offline Demo-Daten als Backup verwenden", value=False)

fallback_auto = False
if demo_modus:
    wetter = pd.read_csv(BASE_DIR / "demo_wetter.csv", index_col=0, parse_dates=True)
    quelle = "Demo-Daten (manuell gewählt)"
else:
    try:
        wetter = hole_wetter(gewaehlter_tag)
        quelle = "Live-Wettervorhersage (aus Open-Meteo API)"
    except Exception:
        wetter = pd.read_csv(BASE_DIR / "demo_wetter.csv", index_col=0, parse_dates=True)
        quelle = "Demo-Daten (Backup für Live-Abruf fehlgeschlagen)"
        fallback_auto = True

prognose = np.clip(modell.predict(baue_features(wetter)), 0, None)
erg = pd.DataFrame({"kWh": prognose}, index=wetter.index).tz_convert("Europe/Berlin")
x_berlin = erg.index # selbe Zeitachse 
datum_label = f"{x_berlin[0]:%d.%m.%Y}"

tagesertrag   = float(prognose.sum())
haushalt_tage = tagesertrag / KWH_HAUSHALT_TAG

if fallback_auto:
    st.warning("⚠️ Live-Abruf für diesen Tag nicht möglich – angezeigt werden feste "
               "Demo-Daten (ein Beispieltag), die sich **nicht** mit dem Datum ändern.")

st.write("---")
st.subheader("☀️ Der Treiber: die Sonnenstrahlung")
st.caption("Verlauf der Globalstrahlung")

fig_strahlung = go.Figure(go.Scatter(
    x=x_berlin, y=wetter["shortwave_radiation"].values, mode="lines",
    line=dict(color="#DAA520", width=3.5),
    fill="tozeroy", fillcolor="rgba(218, 165, 32, 0.15)"))
fig_strahlung.update_layout(
    template="plotly_white",
    title=dict(text="Voraussichtliche Globalstrahlung auf Bodenniveau", font=dict(size=22, color="#000000")),
    font=dict(size=16, color="#000000"),
    xaxis=dict(title=dict(text="Uhrzeit", font=dict(size=18, color="#000000")),
               tickfont=dict(size=14, color="#000000"), linecolor="#000000", linewidth=1),
    yaxis=dict(title=dict(text="Einstrahlung (W/m²)", font=dict(size=18, color="#000000")),
               tickfont=dict(size=14, color="#000000"), linecolor="#000000", linewidth=1),
    height=420, margin=dict(l=10, r=10, t=50, b=10),
    plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig_strahlung, use_container_width=True)

st.write("---")
st.subheader("📈 Die Prognose: der Stromertrag")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Tagesertrag (Prognose)", f"{tagesertrag:.1f} kWh")
k2.metric("Versorgt einen Haushalt", f"{haushalt_tage:.1f} Tage")
k3.metric("Spitzenstunde", f"{erg['kWh'].idxmax():%H:%M} Uhr")
k4.metric("Spitzen-Ertrag (1 h)", f"{prognose.max():.1f} kWh")

fig = go.Figure(go.Scatter(
    x=x_berlin, y=erg["kWh"], mode="lines",
    line=dict(color="steelblue", width=3.5),
    fill="tozeroy", fillcolor="rgba(70, 130, 180, 0.2)"))
fig.update_layout(
    template="plotly_white",
    title=dict(text=f"So viel Strom erwartet unser Modell am {datum_label}", font=dict(size=24, color="#000000")),
    font=dict(size=16, color="#000000"),
    xaxis=dict(title=dict(text="Uhrzeit", font=dict(size=20, color="#000000")),
               tickfont=dict(size=16, color="#000000"), linecolor="#000000", linewidth=1),
    yaxis=dict(title=dict(text="Ertrag (kWh)", font=dict(size=20, color="#000000")),
               tickfont=dict(size=16, color="#000000"), linecolor="#000000", linewidth=1, range=[0, 30]),
    height=480, margin=dict(l=10, r=10, t=60, b=10),
    plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)

st.write("---")
st.subheader(f"🔌 Was steckt in diesen {tagesertrag:.0f} kWh?")
st.caption("Damit könnte man an diesem Tag theoretisch …")

v1, v2, v3, v4, = st.columns(4)
v1.metric("🧺 Waschmaschine", f"{de(tagesertrag / KWH_WASCHMASCHINE)}×", help="ca. 0,8 kWh je 60°C-Ladung")
v2.metric("🍽️ Geschirrspüler", f"{de(tagesertrag / KWH_GESCHIRR)}×", help="ca. 1,2 kWh je Spülgang")
v3.metric("🚗 E-Auto fahren", f"{de(tagesertrag / KWH_EAUTO_KM)} km", help="ca. 16 kWh / 100 km")
v4.metric("Strom-Gegenwert", f"{tagesertrag * 0.37:.2f} €", help="grob mit 0,37 €/kWh – nur Größenordnung, abhängig von Eigenverbrauch/Einspeisung")

st.caption("Veranschaulichung der **erzeugten** Strommenge anhand von Durchschnittswerten – "
           "keine Aussage über den tatsächlichen Verbrauch der Anlage.")

st.write("---")
st.caption(f"Datenquelle: {quelle}")
st.info("Hinweis: Das Modell wurde nur auf vorhandenen echten Daten September 2025 bis April 2026 trainiert.")

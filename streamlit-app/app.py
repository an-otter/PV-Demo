from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.graph_objects as go
from datetime import date, timedelta

# Pfad absolut absichern
BASE_DIR = Path(__file__).resolve().parent 

LAT, LON = 52.22, 13.20
STROMPREIS = 0.37  # €/kWh
WETTER_VARS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
               "temperature_2m", "relative_humidity_2m", "cloud_cover"]

st.set_page_config(page_title="PV-Ertragsprognose", page_icon="☀️", layout="wide")

@st.cache_resource
def lade_modell():
    # KORREKTUR 1: BASE_DIR eingebaut
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

st.title("📈 PhotoVoltaik Vorhersage")
st.caption("Stündliche Ertragsprognose · Modell: Gradient Boosting, Variante B (nur reale Messdaten)")

heute = date.today()
links, rechts = st.columns([1, 3])
with links:
    gewaehlter_tag = st.date_input("Prognosetag", value=heute + timedelta(days=1),
                                   min_value=heute - timedelta(days=70),
                                   max_value=heute + timedelta(days=14))
    demo_modus = st.checkbox("offline Demo-Daten als Backup verwenden", value=False)

fallback_auto = False
if demo_modus:
    # KORREKTUR 2: BASE_DIR eingebaut
    wetter = pd.read_csv(BASE_DIR / "demo_wetter.csv", index_col=0, parse_dates=True)
    quelle = "Demo-Daten (manuell gewählt)"
else:
    try:
        wetter = hole_wetter(gewaehlter_tag)
        quelle = "Live-Wettervorhersage (aus Open-Meteo API)"
    except Exception:
        # KORREKTUR 3: BASE_DIR eingebaut
        wetter = pd.read_csv(BASE_DIR / "demo_wetter.csv", index_col=0, parse_dates=True)
        quelle = "Demo-Daten (Backup für Live-Abruf fehlgeschlagen)"
        fallback_auto = True

prognose = np.clip(modell.predict(baue_features(wetter)), 0, None)
erg = pd.DataFrame({"kWh": prognose}, index=wetter.index).tz_convert("Europe/Berlin")
tagesertrag = prognose.sum()
euro_wert = tagesertrag * STROMPREIS

with rechts:
    if fallback_auto:
        st.warning("⚠️ Live-Abruf für diesen Tag nicht möglich – angezeigt werden feste "
                   "Demo-Daten (ein Beispieltag), die sich **nicht** mit dem Datum ändern.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tagesertrag (Prognose)", f"{tagesertrag:.1f} kWh")
    k2.metric("Strom-Gegenwert", f"{euro_wert:.2f} €")
    k3.metric("Spitzenstunde", f"{erg['kWh'].idxmax():%H:%M} Uhr")
    k4.metric("Spitzenleistung", f"{prognose.max():.1f} kWh/h")
    st.caption(f"Strom-Gegenwert grob mit {STROMPREIS:.2f} €/kWh gerechnet – nur als Größenordnung "
               "und Gedankenstütze. Der tatsächliche Wert hängt von Eigenverbrauch und Einspeisung ab.")

fig = go.Figure(go.Scatter(
    x=erg.index, y=erg["kWh"], mode="lines",
    line=dict(color="steelblue", width=3.5), # Sattes Steelblue für Modell B
    fill="tozeroy", fillcolor="rgba(70, 130, 180, 0.2)"))

fig.update_layout(
    template="plotly_white",
    title=dict(text="Stündlicher PV-Ertrag (Uhrzeit)", font=dict(size=24, color="#000000")), # Titel Schwarz & größer
    font=dict(size=16, color="#000000"), # Globale Schrift auf Schwarz
    
    # X-Achse: Titel, Zahlen und die feine Achsenlinie komplett Schwarz
    xaxis=dict(
        title=dict(text="Uhrzeit", font=dict(size=20, color="#000000")), 
        tickfont=dict(size=16, color="#000000"),
        linecolor="#000000",
        linewidth=1
    ),
    
    # Y-Achse: Titel, Zahlen und die feine Achsenlinie komplett Schwarz + starrer Bereich
    yaxis=dict(
        title=dict(text="Ertrag (kWh/h)", font=dict(size=20, color="#000000")), 
        tickfont=dict(size=16, color="#000000"),
        linecolor="#000000",
        linewidth=1,
        range=[0, 30]
    ),
    
    height=480, 
    margin=dict(l=10, r=10, t=60, b=10),
    plot_bgcolor="white", 
    paper_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

# DIAGRAMM 2: GLOBALSTRAHLUNG (NEU DRUNTER)
# ==========================================
st.write("---") # Trennlinie für optische Sauberkeit
st.subheader("☀️ Der Modell-Treiber: Prognostizierte Solarstrahlung")
st.caption("Verlauf der Globalstrahlung (Kurzwellige Einstrahlung) aus den Wetterdaten für denselben Zeitraum.")

# In der API heißt das Feld 'shortwave_radiation' -> das ist die Globalstrahlung
fig_strahlung = go.Figure(go.Scatter(
    x=wetter.index, y=wetter["shortwave_radiation"], mode="lines",
line=dict(color="#DAA520", width=3.5), # "Goldenrod" / Dunkles Gold für maximalen Beamer-Kontrast
    fill="tozeroy", fillcolor="rgba(218, 165, 32, 0.15)"))

fig_strahlung.update_layout(
    template="plotly_white",
    title=dict(text="Prognostizierte Globalstrahlung auf Bodenniveau", font=dict(size=22, color="#000000")),
    font=dict(size=16, color="#000000"),
    xaxis=dict(
        title=dict(text="Uhrzeit", font=dict(size=18, color="#000000")), 
        tickfont=dict(size=14, color="#000000"),
        linecolor="#000000", linewidth=1
    ),
    yaxis=dict(
        title=dict(text="Einstrahlung (W/m²)", font=dict(size=18, color="#000000")), 
        tickfont=dict(size=14, color="#000000"),
        linecolor="#000000", linewidth=1
        # Keine feste Range, da Strahlung im Sommer weit über 800 W/m² gehen kann
    ),
    height=420, margin=dict(l=10, r=10, t=50, b=10),
    plot_bgcolor="white", paper_bgcolor="white"
)

st.plotly_chart(fig_strahlung, use_container_width=True)

st.caption(f"Datenquelle: {quelle}")
st.info("Hinweis: Das Modell wurde nur auf vorhandenen echten Daten September 2025 bis April 2026 trainiert.")
"""Streamlit app for ODS prediction using deployed winner model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference_pipeline import predict_text


APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"


def _read_svg(asset_name: str) -> str:
    path = ASSETS_DIR / asset_name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


LOGO_SHIELD_SVG = _read_svg("uniandes_logo.svg")
LOGO_WORDMARK_SVG = _read_svg("uniandes_wordmark.svg")

st.set_page_config(page_title="Clasificador ODS", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');

    :root {
        --bg: #f7f8fa;
        --card: #ffffff;
        --text: #111317;
        --muted: #58616c;
        --accent: #121417;
        --andes-yellow: #f4e000;
        --border: #e7eaee;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', 'Avenir Next', 'Segoe UI', sans-serif;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(circle at 0% 0%, #fffbe5 0%, transparent 30%),
            radial-gradient(circle at 100% 0%, #f0f2f5 0%, transparent 34%),
            var(--bg);
    }

    .block-container {
        max-width: 860px;
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    .uniandes-band {
        background: #111417;
        border: 1px solid #222b31;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 0.9rem;
    }

    .uniandes-band .logo-shield svg {
        height: 60px;
        width: auto;
        display: block;
    }

    .uniandes-band .logo-wordmark svg {
        height: 44px;
        width: auto;
        display: block;
    }

    .uniandes-fallback {
        color: #fff;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .app-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.15rem 1.1rem 1rem 1.1rem;
        box-shadow: 0 8px 22px rgba(17, 20, 23, 0.05);
        margin-bottom: 1rem;
    }

    .app-kicker {
        margin: 0;
        color: #3c434c;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
    }

    .app-title {
        margin: 0.45rem 0 0 0;
        font-size: 1.46rem;
        font-weight: 700;
        letter-spacing: -0.015em;
    }

    .app-subtitle {
        margin: 0.55rem 0 0 0;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .course-tag {
        display: inline-block;
        margin-top: 0.7rem;
        padding: 0.24rem 0.52rem;
        border-radius: 8px;
        background: #fff9cc;
        border: 1px solid #efe4a2;
        color: #3a3720;
        font-size: 0.78rem;
        font-weight: 600;
    }

    [data-testid="stTextArea"] textarea {
        border-radius: 12px;
        border: 1px solid var(--border);
        background: #fcfcfd;
        font-size: 0.96rem;
        line-height: 1.55;
    }

    [data-testid="stTextArea"] textarea:focus {
        box-shadow: 0 0 0 0.13rem rgba(18, 20, 23, 0.12);
        border-color: rgba(18, 20, 23, 0.52);
    }

    div.stButton > button {
        height: 2.7rem;
        border-radius: 10px;
        border: 1px solid var(--accent);
        background: var(--accent);
        color: #fff;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background: #000;
        border-color: #000;
    }

    .meta-line {
        color: var(--muted);
        font-size: 0.86rem;
        margin-top: -0.2rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    @media (max-width: 720px) {
        .uniandes-band {
            flex-direction: column;
            align-items: flex-start;
        }
        .uniandes-band .logo-wordmark svg {
            height: 36px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if LOGO_SHIELD_SVG or LOGO_WORDMARK_SVG:
    logos_html = (
        "<div class='uniandes-band'>"
        + (f"<div class='logo-shield'>{LOGO_SHIELD_SVG}</div>" if LOGO_SHIELD_SVG else "")
        + (f"<div class='logo-wordmark'>{LOGO_WORDMARK_SVG}</div>" if LOGO_WORDMARK_SVG else "")
        + "</div>"
    )
else:
    logos_html = "<div class='uniandes-band'><p class='uniandes-fallback'>Universidad de los Andes, Colombia</p></div>"

st.markdown(logos_html, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-card">
        <p class="app-kicker">Microproyecto 2 · Clasificación de Texto</p>
        <h1 class="app-title">Clasificador de Objetivos de Desarrollo Sostenible</h1>
        <p class="app-subtitle">
            Ingresa un texto y obtén la predicción del ODS con el modelo ganador,
            junto con su nivel de confianza y el ranking de clases más probables.
        </p>
        <span class="course-tag">
            Proyecto que forma parte del curso de Aprendizaje No Supervisado de la Maestría en Inteligencia Artificial.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("#### Texto a clasificar")
text_input = st.text_area(
    "Texto",
    label_visibility="collapsed",
    height=190,
    placeholder="Ejemplo: Programa educativo comunitario para reducir brechas en acceso a internet en zonas rurales.",
)

top_k = 1
predict_clicked = st.button("Predecir ODS", type="primary", use_container_width=True)

if predict_clicked:
    if not text_input.strip():
        st.warning("Ingresa un texto antes de predecir.")
    else:
        try:
            with st.spinner("Ejecutando inferencia..."):
                result = predict_text(text_input, top_k=top_k, prefer_deploy=True)

            pred_col, conf_col = st.columns(2)
            with pred_col:
                st.metric("ODS predicho", str(result["predicted_ods"]))
            with conf_col:
                confidence = result.get("confidence")
                if confidence is None:
                    st.metric("Confianza", "N/A")
                else:
                    st.metric("Confianza", f"{confidence:.2%}")

            model_name = result.get("model_name", "N/A")
            st.markdown(f"<p class='meta-line'>Modelo: <strong>{model_name}</strong></p>", unsafe_allow_html=True)
            if result.get("artifact_model_path"):
                st.caption(f"Artifact: `{result['artifact_model_path']}`")

            if result.get("top_k"):
                df_top = pd.DataFrame(result["top_k"]).rename(columns={"ods": "ODS", "prob": "Probabilidad"})
                if "Probabilidad" in df_top.columns:
                    df_top["Probabilidad"] = (df_top["Probabilidad"] * 100).map(lambda x: f"{x:.2f}%")
                st.markdown("#### Top clases")
                st.dataframe(df_top, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"No se pudo realizar la predicción: {exc}")

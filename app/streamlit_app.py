import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import io

# Configuration
st.set_page_config(
    page_title="EO Change Detection",
    page_icon="🔭",
    layout="wide"
)

OSCD_DIR = r"C:\Users\jeane\Documents\Alternance\eo-change-detection\data\OSCD"
RESULTS_PATH = r"C:\Users\jeane\Documents\Alternance\eo-change-detection\data\results\change_detection_results.json"

@st.cache_resource
def load_model():
    model_name = "flax-community/clip-rsicd-v2"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return model, processor

@st.cache_data
def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)

def load_rgb(city_dir, imgs_folder):
    folder = os.path.join(city_dir, imgs_folder)
    b4 = rasterio.open(os.path.join(folder, "B04.tif")).read(1).astype(np.float32)
    b3 = rasterio.open(os.path.join(folder, "B03.tif")).read(1).astype(np.float32)
    b2 = rasterio.open(os.path.join(folder, "B02.tif")).read(1).astype(np.float32)
    img = np.stack([b4, b3, b2], axis=0)
    valid = img[img > 0]
    if len(valid) > 0:
        p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
        img = np.clip((img - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))

def encode_image(img_array, model, processor):
    img_pil = Image.fromarray(img_array)
    inputs = processor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        emb = model.visual_projection(outputs.pooler_output)
        emb = F.normalize(emb, dim=-1)
    return emb.numpy()[0]

# Interface
st.title("🔭 EO Change Detection")
st.markdown("Détection de changement sémantique sur images Sentinel-2 via embeddings RemoteCLIP")

# Chargement
model, processor = load_model()
results = load_results()
df = pd.DataFrame(results).sort_values("change_score", ascending=False)

# Vue globale
st.subheader("🌍 Vue globale — Score de changement par ville")

col1, col2, col3 = st.columns(3)
col1.metric("Villes analysées", len(df))
col2.metric("Changement max", f"{df['change_score'].max():.4f}")
col3.metric("Ville la plus changée", df.iloc[0]["city"].upper())

st.divider()

# Carte des scores
st.subheader("📊 Classement des villes")
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#E24B4A" if s > 0.05 else "#378ADD" for s in df["change_score"]]
ax.barh(df["city"], df["change_score"], color=colors)
ax.axvline(x=0.05, color="gray", linestyle="--", alpha=0.5, label="Seuil changement modéré")
ax.set_xlabel("Score de changement (1 - cosine similarity)")
ax.set_title("Score de changement sémantique par ville")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

st.divider()

# Analyse détaillée par ville
st.subheader("🔍 Analyse détaillée")

city_names = [r["city"] for r in results]
selected_city = st.selectbox("Sélectionne une ville", sorted(city_names))

if selected_city:
    city_result = next(r for r in results if r["city"] == selected_city)
    city_dir = os.path.join(OSCD_DIR, selected_city)

    col1, col2, col3 = st.columns(3)
    col1.metric("Score de changement", f"{city_result['change_score']:.4f}")
    col2.metric("Interprétation", city_result["interpretation"])
    col3.metric("Période", f"{city_result['date_before']} → {city_result['date_after']}")

    # Images avant/après
    with st.spinner("Chargement des images..."):
        img_before = load_rgb(city_dir, "imgs_1_rect")
        img_after = load_rgb(city_dir, "imgs_2_rect")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_before, caption=f"Avant — {city_result['date_before']}", use_column_width=True)
    with col2:
        st.image(img_after, caption=f"Après — {city_result['date_after']}", use_column_width=True)

    # Types de changement détectés
    st.subheader("🏷️ Types de changement détectés")
    for i, change in enumerate(city_result["top_changes"]):
        medal = ["🥇", "🥈", "🥉"][i]
        st.write(f"{medal} **{change['description']}** — similarité : {change['score']:.4f}")

    # Carte GPS
    if city_result["coords"]:
        st.subheader("🗺️ Localisation")
        lon, lat = city_result["coords"]
        df_map = pd.DataFrame([{"lat": lat, "lon": lon}])
        st.map(df_map)

st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Modèle", "RemoteCLIP")
col2.metric("Dataset", "OSCD — 24 villes")
col3.metric("Métrique", "1 - cosine similarity")
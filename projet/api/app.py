import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# from projet.ml_logic.model import predict_image  # À activer plus tard

st.set_page_config(page_title="Détection de déchets", layout="wide")
st.title("🌊 Détection de déchets dans l’eau")

tabs = st.tabs(["📷 Caméra en direct", "📁 Image", "ℹ️ Classes", "❓À propos"])

# === 1. Onglet Caméra en direct ===
with tabs[0]:
    st.subheader("📷 Détection via webcam")

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    if "run_once" not in st.session_state:
        st.session_state.run_once = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Lancer la caméra"):
            st.session_state.camera_active = True
            st.session_state.run_once = False
    with col2:
        if st.button("⏹️ Arrêter la caméra"):
            st.session_state.camera_active = False

    stframe = st.empty()

    if st.session_state.camera_active and not st.session_state.run_once:
        st.session_state.run_once = True
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Impossible d’ouvrir la caméra.")
            st.session_state.camera_active = False
        else:
            st.info("Caméra activée. Appuyez sur 'Arrêter la caméra' pour couper.")
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Erreur de lecture de la caméra.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # predict_image(frame_rgb) par ex.
                # prediction = predict_image(frame_rgb)
                # st.write("Détection :", prediction)

                stframe.image(frame_rgb, channels="RGB")
                time.sleep(0.03)

            cap.release()
            stframe.empty()
            st.success("✅ Caméra arrêtée.")

# === 2. Onglet Image ===
with tabs[1]:
    st.subheader("📁 Charger une image")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargée", use_column_width=True)

        st.markdown("Ajout possible de prédiction sur image ici :")
        prediction = predict_image(np.array(image))
        st.write("Détection :", prediction)

# === 3. Onglet Classes ===
with tabs[2]:
    st.subheader("📦 Classes reconnues")
    st.markdown("""
    Le modèle reconnaît les types de déchets suivants :
    - ♳ Plastique
    - ♴ Métal
    - ♵ Verre
    """)

# === 4. Onglet À propos ===
with tabs[3]:
    st.subheader("❓ À propos")
    st.markdown("""
    Ce projet vise à détecter automatiquement les déchets dans les eaux marines à partir d’images ou de flux vidéo.

    **Développé par Hadrien Touchon le roi de ce monde.**
    """)

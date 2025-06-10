import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import base64
import requests

BASE_URL = "http://localhost:8000"

# from projet.ml_logic.model import predict_image  # À activer plus tard

st.set_page_config(
    page_title="Pour des eaux claires, wall-e fait la guerre aux déchets en mer.",
    layout="wide"
)

# === CSS : fond et boutons arrondis ===
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #00497f, #000000);
        background-attachment: fixed;
        color: white;
    }

    .block-container {
        background-color: rgba(0, 0, 0, 0);
    }

    .nav-button {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid white;
        border-radius: 999px;
        padding: 10px 20px;
        margin-right: 10px;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    .nav-button:hover {
        background-color: rgba(255, 255, 255, 0.25);
    }

    .nav-button-selected {
        background-color: rgba(255, 255, 255, 0.3);
        border: 2px solid white;
    }
    </style>
""", unsafe_allow_html=True)

# === Titre
st.title("Pour des eaux claires, wall-e fait la guerre aux déchets en mer.")

# === Menu de navigation personnalisé
tabs = ["Image", "Caméra en direct", "Classes", "À propos"]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = tabs[0]
    # st.subheader("📷 Détection via webcam")

    # if "camera_active" not in st.session_state:
    #     st.session_state.camera_active = False
    # if "run_once" not in st.session_state:
    #     st.session_state.run_once = False

    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("▶️ Lancer la caméra"):
    #         st.session_state.camera_active = True
    #         st.session_state.run_once = False
    # with col2:
    #     if st.button("⏹️ Arrêter la caméra"):
    #         st.session_state.camera_active = False

    # stframe = st.empty()

    # if st.session_state.camera_active and not st.session_state.run_once:
    #     st.session_state.run_once = True
    #     cap = cv2.VideoCapture(0)

    #     if not cap.isOpened():
    #         st.error("❌ Impossible d’ouvrir la caméra.")
    #         st.session_state.camera_active = False
    #     else:
    #         st.info("Caméra activée. Appuyez sur 'Arrêter la caméra' pour couper.")
    #         while st.session_state.camera_active:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 st.warning("Erreur de lecture de la caméra.")
    #                 break

    #             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #             # predict_image(frame_rgb) par ex.
    #             # prediction = predict_image(frame_rgb)
    #             # st.write("Détection :", prediction)

    #             stframe.image(frame_rgb, channels="RGB")
    #             time.sleep(0.03)

    #         cap.release()
    #         stframe.empty()
    #         st.success("✅ Caméra arrêtée.")


    gif_path = os.path.join(os.path.dirname(__file__), "0609.gif")
    with open(gif_path, "rb") as file_:
        contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")

    st.markdown(
        f'''
        <div style="display: flex; justify-content: center; align-items: center; height: 90vh;">
            <img src="data:image/gif;base64,{data_url}" alt="cat gif" style="max-width: 100vw; max-height: 85vh; width: 100vw; height: 85vh; object-fit: contain; display: block;"/>
        </div>
        ''',
        unsafe_allow_html=True,
    )

elif st.session_state.active_tab == "Image":
    st.subheader("Choisir un modèle")
    model = st.radio('Choisir un modèle', ('CNN', 'Yolo'), 1)
    if model == "CNN":
        model_name = "olympe_model"
    elif model == "Yolo":
        model_name = "yolo"
    requests.get(BASE_URL + "/model", params={
        "model_name": model_name
    })
    st.subheader("📁 Charger une image")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargée", use_column_width=True)


        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        url_post = BASE_URL+"/predict"
        response = requests.post(url_post, files=files)

        json = response.json()

        if model == "CNN":
            pass
        elif model == "Yolo":
            for idc, waste_category_idx in enumerate(json["waste_categories"]):
                st.write(waste_category_idx)
                st.write(json["confidence_score"][idc])

        st.write(response.json())

        st.markdown("Ajout possible de prédiction sur image ici :")
        # prediction = predict_image(np.array(image))
        # st.write("Détection :", prediction)

elif st.session_state.active_tab == "Classes":
# === 3. Onglet Classes ===
    st.subheader("📦 Classes reconnues")
    st.markdown("""
    Le modèle reconnaît les types de déchets suivants :
    - ♳ Plastique
    - ♴ Métal
    - ♵ Verre
    """)

elif st.session_state.active_tab == "À propos":
    st.subheader("❓ À propos")
    st.markdown("""
    Ce projet vise à détecter automatiquement les déchets dans les eaux marines à partir d’images ou de flux vidéo.

    **Développé par Hadrien Touchon le roi de ce monde.**
    """)

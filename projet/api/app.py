import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import base64
import requests

BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Pour des eaux claires, wall-e fait la guerre aux déchets en mer.",
    layout="wide"
)

# === Titre
st.title("Pour des eaux claires, wall-e fait la guerre aux déchets en mer.")

# === Menu de navigation personnalisé
tabs = st.tabs(["Caméra en direct", "Image", "Classes", "À propos"])

with tabs[0]:
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


with tabs[1]:
    st.subheader("Choisir un modèle")
    model = st.radio('Choisir un modèle', ('CNN', 'Yolo'), 1)
    model_name = "olympe_model" if model == "CNN" else "yolo"
    response = requests.get(BASE_URL + "/model", params={"model_name": model_name})
    if response.status_code == 200:
        st.success(f"Modèle {model_name} activé")
    else:
        st.error("Erreur lors de l’activation du modèle")

    st.subheader("📁 Charger une image")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargée", use_column_width=True)
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        url_post = BASE_URL + "/predict"
        response = requests.post(url_post, files=files)

        if response.status_code == 200:
            json = response.json()
            if model == "CNN":
                classes = ["Verre", "Métal", "Plastique"]
                predicted_class = classes[np.argmax(json["prediction"])]
                confidence = np.max(json["prediction"])
                st.success(f"Classe prédite : {predicted_class} avec une confiance de {confidence:.2f}")
            elif model == "Yolo":
                for idc, waste_category_idx in enumerate(json["waste_categories"]):
                    st.write(f"Classe : {waste_category_idx} - Confiance : {json['confidence_score'][idc]:.2f}")
                st.write(json)
        else:
            st.error("Erreur lors de la prédiction")

with tabs[2]:
# === 3. Onglet Classes ===
    st.subheader("📦 Classes reconnues")
    st.markdown("""
    Le modèle reconnaît les types de déchets suivants :
    - ♳ Plastique
    - ♴ Métal
    - ♵ Verre
    """)

with tabs[3]:
    st.subheader("❓ À propos")
    st.markdown("""
    Ce projet vise à détecter automatiquement les déchets dans les eaux marines à partir d’images ou de flux vidéo.

    **Développé par Hadrien Touchon le roi de ce monde.**
    """)

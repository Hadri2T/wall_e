import streamlit as st
from PIL import Image
import os

st.title("♻️ Détection de déchets marins ♻️ ")
st.markdown("## Uploadez une image pour détecter le type de déchet")


uploaded_file = st.file_uploader("Glissez une image ou cliquez pour uploader", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Affichage de l’image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image uploadée", use_column_width=True)

    # Simuler une prédiction pour l’instant
    st.markdown("### Résultat de la prédiction :")
    st.success("✅ Classe détectée : plastique")

    # Sauvegarder temporairement
    temp_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    image.save(temp_path)

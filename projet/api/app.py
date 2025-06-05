import streamlit as st
from PIL import Image
import os
import zipfile


#Nom de la page sur l'onglet + logo
st.set_page_config(page_title="Détection de déchets marins", page_icon="🌊", layout="centered")


st.title("♻️ Détection de déchets marins ♻️ ") #Titre
st.markdown("### Uploadez une image pour détecter le type de déchet")


uploaded_file = st.file_uploader(
    "Glissez une image *ou* un fichier `.zip` contenant plusieurs images",
    type=["png", "jpg", "jpeg", "zip"]
)


if uploaded_file is not None:
    if uploaded_file.name.endswith(".zip"):
        st.markdown("### 📂 Contenu du fichier ZIP :")
        with zipfile.ZipFile(uploaded_file) as archive:
            image_files = [f for f in archive.namelist() if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            if not image_files:
                st.warning("Aucune image valide trouvée dans le ZIP.")
            else:
                for filename in image_files:
                    with archive.open(filename) as file:
                        image = Image.open(file)
                        st.image(image, caption=filename, use_column_width=True)
                        st.success(f"✅ Classe détectée : plastique")

    else:
        # Cas standard : image seule
        image = Image.open(uploaded_file)
        st.image(image, caption="Image uploadée", use_container_width=True)
        st.markdown("### Résultat de la prédiction :")
        st.success("✅ Classe détectée : plastique")

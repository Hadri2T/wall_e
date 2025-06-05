<<<<<<< HEAD
def undersample_class(df, target_class='Plastic', frac=0.2, class_col='class', random_state=42):
    """
    Sous-échantillonne une classe spécifique dans un DataFrame pour équilibrer les classes.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        target_class (str): La classe à sous-échantillonner.
        frac (float): La fraction à conserver pour la classe cible.
        class_col (str): Le nom de la colonne des classes.
        random_state (int): Graine pour la reproductibilité.

    Returns:
        pd.DataFrame: Un DataFrame équilibré avec la classe cible sous-échantillonnée.
    """
    # Séparer les lignes correspondant à la classe cible et aux autres classes
    df_target = df[df[class_col] == target_class]
    df_other = df[df[class_col] != target_class]

    # Sous-échantillonnage de la classe cible
    df_target_sampled = df_target.sample(frac=frac, random_state=random_state)

    # Concaténer les deux DataFrames et mélanger les lignes
    df_balanced = pd.concat([df_target_sampled, df_other]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_balanced

def class_proportion(df, col='class'):
    """
    Calcule la proportion de chaque classe dans une colonne spécifiée d'un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        col (str, optional): Le nom de la colonne pour laquelle calculer la proportion des classes.
            Par défaut 'class'.

    Returns:
        pd.Series: Une série contenant la proportion (en pourcentage) de chaque classe dans la colonne spécifiée.

    Exemple:
        >>> class_proportion(df, col='target')
        0    60.0
        1    40.0
        Name: target, dtype: float64
    """
    prop = df[col].value_counts(normalize=True) * 100
    return prop

class_proportion(df)
=======
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image

#Filtrer les images selon le nombres d'objets

def filter_images(csv_path, min_objects=1, max_objects=None):
#csv_path = chemin vers le csv --> str
#min_objects = nombre minimum d'objets à garder --> int
#max_object = nomber maximum d'objets à garder --> int

    df = pd.read_csv(csv_path)

    #Compter le nombre d'occurence des noms des images
    count = df["filename"].value_counts()

    # Filtrer selon les bornes min/max
    if max_objects is not None:
        valid_filenames = count[(count >= min_objects) & (count <= max_objects)].index
    else:
        valid_filenames = count[count >= min_objects].index

    # Garder uniquement les lignes correspondantes
    filtered_df = df[df['filename'].isin(valid_filenames)]

    return filtered_df



#Redimensionner les images avec du padding si nécessaire
def resize_with_padding(image_path, target_size=(64, 64)):

    #print(image_path)

    #On lis le contenu du path
    image_file = tf.io.read_file(image_path)

    #print(image_file)

    #On transforme ce qu'on lit en image
    image = tf.image.decode_jpeg(image_file, channels=3)

    #print(image)

    #Ca marche pas avec des floats au dessus de 1
    image = tf.image.convert_image_dtype(image, tf.float32)

    #tf.shape - Returns a tensor containing the shape of the input tensor.
    image_shape = tf.shape(image)

    #print(image_shape)

    #Trouver hauteur et largeur de l'image
    height = image_shape[0]
    width = image_shape[1]

    #Trouver la taille de la plus grande dimension, pour que ce soit la taille du carré que l'on va utiliser pour le padding
    max_dim = tf.maximum(height, width)

    #On calcule combien de pixels ajouter en haut/bas et à gauche/droite, pour que l’image devienne carrée, centrée dans le carré.
    #On divise par deux puisque il y a ura la moitié au dessus et la moitié en dessous
    pad_height = (max_dim - height) // 2
    pad_width = (max_dim - width) // 2

    #On positionne l'image dans un cadre plus grand
    #On met un certain nombre de pixels en haut/bas ou droite/gauche, dépendant de ce qu'on a calculer avant
    padded_image = tf.image.pad_to_bounding_box(
        image, pad_height, pad_width, max_dim, max_dim
    )

    #On redimensionne la taille finale
    final_image = tf.image.resize(padded_image, target_size)

    return final_image


#Resize les bounding boxes comme j'ai resize les images
def resize_bounding_boxes(df, target_size = (64,64)):
    #Prend comme argument le dataframe avec toutes les infos du csv et la target_size
    #Redonne un dataframe avec les datas mise à jour par rapport à la target_size

    new_bounding_boxes = []
    #Liste ou il y a aura les bounding boxes de bonnes taille

    for index, row in df.iterrows():
    # mais on n’utilise pas index je crois

        width, height = row['width'], row['height']
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

        max_dim = max(width, height) #Taille du carré qu'on va obtenir après le padding
        pad_height = (max_dim - height) // 2 #Combien de pixels on ajoute
        pad_width = (max_dim - width) // 2

        #On décale les x/y selon les nouvelles dimensions
        xmin_padded = xmin + pad_width
        xmax_padded = xmax + pad_width
        ymin_padded = ymin + pad_height
        ymax_padded = ymax + pad_height


        #Calculer le facteur d’échelle entre la taille de l’image paddée  et la taille finale souhaitée
        #Comme ça on sait de combien on augmente ou diminue chaque pixel
        scale = target_size[0] / max_dim

        #On applique le scale aux données paddées
        xmin_resized = xmin_padded * scale
        xmax_resized = xmax_padded * scale
        ymin_resized = ymin_padded * scale
        ymax_resized = ymax_padded * scale

        #On enregistre les résultats de cette ligne sous forme de dictionnaire, qu’on ajoute à la liste new_boxes.
        new_bounding_boxes.append({
            'filename': row['filename'],
            'class': row['class'],
            'xmin': xmin_resized,
            'ymin': ymin_resized,
            'xmax': xmax_resized,
            'ymax': ymax_resized
        })

    df_bounding_boxes = pd.DataFrame(new_bounding_boxes)

    return df_bounding_boxes


def preprocess_and_save_dataset(
    csv_path,
    image_folder,
    preprocessed_images_root,
    target_size=(64, 64),
    gcp=False
):
    ''' csv_path (str) : chemin vers le fichier d’annotations

    image_folder (str) : dossier contenant les images originales

    preprocessed_images (str) : racine du dossier de sortie

    target_size (tuple) : dimension finale (par défaut (64, 64))

    gcp (bool) : pour uploader vers GCS '''

    df = pd.read_csv(csv_path)


    # ten= df["filename"].unique()[:10]

    # df_test = df[df["filename"].isin(ten)]

    df_resized = resize_bounding_boxes(df, target_size)

    size_file = str(target_size[0])
    split_name = os.path.basename(image_folder.rstrip("/"))
    output_dir = os.path.join(preprocessed_images_root, f"preprocessed_images_{size_file}_{split_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Boucler sur chaque image unique dans le CSV
    for filename in df_resized['filename'].unique():
        #Construction du chemin complet vers l'image
        image_path = os.path.join(image_folder, filename)

        try:
            # Redimensionner et ajouter du padding à l'image
            image_tensor = resize_with_padding(image_path, target_size)

            # 1. Convertir le tf en np car PIL.Image ne prend que des arrays
            # 2. Faire *255 car on avait mis au format 0 à 1 pour des questions de stabilités, mais que pour sortir des images il faut des couleurs
            # 3. On remet en int car les floats ne sont pas acceptés par PIL
            image_array = (image_tensor.numpy() * 255).astype(np.uint8)

            # Définir le chemin de sortie
            output_path = os.path.join(output_dir, filename)

            if gcp:
                # Je sais pas comment envoyer sur le cloud
                print(f"Image prête pour upload : {filename}")
            else:
                # Sauvegarde locale
                # Image.fromarray transforme l'array en image
                #.save() permet de le save dans le format que je veux en suivant le chemin que je veux
                Image.fromarray(image_array).save(output_path, format="JPEG")

        except Exception as e:
            print(f"Erreur avec {filename} : {e}")

        # Sauvegarder le CSV des bboxes redimensionnées car elles ont changé de coordonnées
    if gcp:
        print("CSV prêt pour upload")
    else:
        csv_output_path = os.path.join(output_dir, f"resized_annotations_{size_file}_{split_name}.csv")
        df_resized.to_csv(csv_output_path, index=False)

    return None
>>>>>>> 4ab36519bd2713f1f54d19188dce1db5e5b83b21

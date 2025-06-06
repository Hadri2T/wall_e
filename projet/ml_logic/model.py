import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import tensorflow
import keras
import keras.callbacks
from keras.callbacks import EarlyStopping
from PIL import Image
import numpy as np
from keras import Sequential, layers, Input


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
# A priori cette fonction va bouger parce que Hadrien s'occupe déjà d'une bonne partie

def load_and_preprocess_images(df, image_dir, img_size=(128, 128)):
    """
    Charge et prétraite les images et les labels à partir d'un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'filename' et 'target'.
        image_dir (str): Dossier contenant les images.
        img_size (tuple): Taille à laquelle redimensionner les images.

    Returns:
        tuple: (X, y) où X est un np.array d'images et y un np.array de labels.
    """
    X = []
    y = []
    for _, row in df.iterrows():
        filepath = os.path.join(image_dir, row['filename'])
        img = Image.open(filepath).convert('RGB').resize(img_size)
        img_array = np.array(img, dtype=np.float32) / 255.  # Normalisation [0,1]
        X.append(img_array)
        y.append(row['encoded_target'])
    X = np.array(X)
    y = np.array(y, dtype=np.float32)
    print(X.shape)
    print(y.shape)
    return X, y # exemple : X = 10567, 64, 64, 3
                          # y = 10567,

# J'ai besoin que ce soit resize + RGB puis converti en nparray + normalisation

def model(X, y, early_stopping, input_shape=(128, 128, 3)):
    """
    Modèle pour trouver classe quand il y a un seul déchet

    Args:
        input_shape (tuple): La forme des images en entrée.

    Returns:
        model (tf.keras.Model): Le modèle compilé.
    """


    model = Sequential([
        Input(shape=(128, 128, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')  # ← 3 classes = plastique, métal, verre
    ])

    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

    model.fit(
        X, y,
        validation_split=0.2,
        batch_size=32,
        epochs=50,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )
    return model

def load_model(model_path):
    """
    Charge le modèle Keras à partir du chemin spécifié.

    Args:
        model_path (str): Chemin vers le modèle Keras.

    Returns:
        model : Le modèle chargé.
    """
    model = None#Olympe c'est pour toi 
    return model

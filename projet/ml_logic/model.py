from PIL import Image
import numpy as np
import os
from tensorflow.keras import Sequential, layers, Input
from tensorflow.keras.callbacks import EarlyStopping




# A priori cette fonction va bouger parce que Hadrien s'occupe déjà d'une bonne partie

def load_and_preprocess_images(df, image_dir, img_size=(64, 64)):
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
        y.append(row['target'])
    X = np.array(X)
    y = np.array(y, dtype=np.float32)
    print(X.shape)
    print(y.shape)
    return X, y # exemple : X = 10567, 64, 64, 3
                          # y = 10567,

# J'ai besoin que ce soit resize + RGB puis converti en nparray + normalisation





def model1(X, y, input_shape=(64, 64, 3), batch_size=32, epochs=50, validation_split=0.2, patience=5):
    """
    Construit et entraîne un modèle CNN pour reconnaître si un déchet est plastique ou non sur un déchet

    Args:
        X (np.array): Images prétraitées.
        y (np.array): Labels.
        input_shape (tuple): Shape des images en entrée.
        batch_size (int): Taille du batch pour l'entraînement.
        epochs (int): Nombre d'époques.
        validation_split (float): Proportion de validation.
        patience (int): Patience pour l'early stopping.

    Returns:
        model: Modèle entraîné.
        history: Historique d'entraînement.
    """
    model = Sequential([
        Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # binaire
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X, y,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)]
    )
    return model, history

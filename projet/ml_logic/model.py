


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
    from tensorflow.keras import Sequential, layers, Input

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

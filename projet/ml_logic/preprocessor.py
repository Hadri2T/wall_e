
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


def undersample_class(df, target_class='Plastic', frac=0.2, class_col='class', random_state=42):
    """
    Sous-échantillonne une classe spécifique dans un DataFrame pour équilibrer les classes.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        target_class (str): La classe à sous-échantillonner.
        frac (float): La fraction à conserver pour la classe cible.
        class_col (str): Le nom de la colonne des classes.
        random_state (int): Graine pour la reproductibilité. Pour éviter d'avoir le même échantillon supprimé à chaque fois

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

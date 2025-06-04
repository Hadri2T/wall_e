def load_data(filepath="raw_data/valid_annotations.csv", max_objects=1):
    """
    Load and preprocess image annotation data from a CSV file.

    This function reads image annotation data, filters images based on the maximum number of objects allowed,
    and creates a binary target column indicating whether the object class is 'Plastic'.

    Args:
        filepath (str): Path to the CSV file containing annotation data. Defaults to "raw_data/valid_annotations.csv".
        max_objects (int): Maximum number of objects allowed per image. Defaults to 1.

    Returns:
        pandas.DataFrame: A DataFrame containing filtered image annotations with an additional 'target' column,
        where 'target' is True if the class is 'Plastic', otherwise False.
    """
    df = filter_images(filepath, max_objects=max_objects)
    df['target'] = (df['class'] == 'Plastic')
    return df

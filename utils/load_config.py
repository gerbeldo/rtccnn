import yaml

def load_config(config_path: str = "../config/base.yaml") -> dict:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str, optional): The file path to the YAML configuration file.

    Returns:
        dict: The configuration settings as a dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

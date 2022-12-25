import  yaml

def get_config():
    """
    Loads config.yaml defined at the root folder and returns a dict object
    """
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        return config
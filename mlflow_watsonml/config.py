import json
import os

from dotenv import load_dotenv


class Config(dict):
    def __init__(self):
        """
        Initializes constants from .env file and environment variables
        """
        super().__init__()

        load_dotenv()

        with open(os.environ.get("WML_CREDENTIALS_FILE"), "r") as f:
            self["wml_credentials"] = json.load(f)

        self["deployment_space_name"] = os.environ.get("DEPLOYMENT_SPACE_NAME")

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

        credentials_file = os.environ.get("WML_CREDENTIALS_FILE")

        if credentials_file is not None and os.path.exists(credentials_file):
            with open(credentials_file, "r") as f:
                self["wml_credentials"] = json.load(f)
        else:
            raise FileNotFoundError(
                "WML Credentials not found. Set `WML_CREDENTIALS_FILE` environment variable"
            )

        self["deployment_space_name"] = os.environ.get("DEPLOYMENT_SPACE_NAME")

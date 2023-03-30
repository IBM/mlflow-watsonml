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

        wml_api_key = os.environ.get("WML_API_KEY")

        location = os.environ.get("LOCATION", "us-south")
        url = f"https://{location}.ml.cloud.ibm.com"

        self["wml_credentials"] = {"apikey": wml_api_key, "url": url}

        self["deployment_space_name"] = os.environ.get("DEPLOYMENT_SPACE_NAME")

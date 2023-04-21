import json
import os
from typing import Dict, Optional

from dotenv import load_dotenv
from mlflow import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class Config(dict):
    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes constants from .env file and environment variables
        """
        super().__init__()

        if config is None:
            load_dotenv()
            config = dict(os.environ)

        # Load the appropriate environment variables based on their presence
        if "WML_CREDENTIALS_FILE" in config and "DEPLOYMENT_SPACE_NAME" in config:
            wml_credentials_file = config["WML_CREDENTIALS_FILE"]

            with open(wml_credentials_file, "r") as f:
                self["wml_credentials"] = json.load(f)

            self["deployment_space_name"] = config["DEPLOYMENT_SPACE_NAME"]

        elif (
            "APIKEY" in config
            and "LOCATION" in config
            and "URL" in config
            and "DEPLOYMENT_SPACE_NAME" in config
        ):
            apikey = config["APIKEY"]
            location = config["LOCATION"]
            url = config["URL"]

            self["wml_credentials"] = {
                "apikey": apikey,
                "url": url,
            }

            self["deployment_space_name"] = config["DEPLOYMENT_SPACE_NAME"]

        elif (
            "TOKEN" in config
            and "LOCATION" in config
            and "URL" in config
            and "DEPLOYMENT_SPACE_NAME" in config
        ):
            token = config["TOKEN"]
            location = config["LOCATION"]
            url = config["URL"]

            self["wml_credentials"] = {
                "token": token,
                "url": url,
            }
            self["deployment_space_name"] = config["DEPLOYMENT_SPACE_NAME"]

        elif (
            "USERNAME" in config
            and "PASSWORD" in config
            and "URL" in config
            and "INSTANCE_ID" in config
            and "VERSION" in config
            and "DEPLOYMENT_SPACE_NAME" in config
        ):
            username = config["USERNAME"]
            password = config["PASSWORD"]
            url = config["URL"]
            instance_id = config["INSTANCE_ID"]
            version = config["VERSION"]

            self["wml_credentials"] = {
                "username": username,
                "password": password,
                "url": url,
                "instance_id": instance_id,
                "version": version,
            }

            self["deployment_space_name"] = config["DEPLOYMENT_SPACE_NAME"]

        elif (
            "USERNAME" in config
            and "APIKEY" in config
            and "URL" in config
            and "INSTANCE_ID" in config
            and "VERSION" in config
            and "DEPLOYMENT_SPACE_NAME" in config
        ):
            username = config["USERNAME"]
            apikey = config["APIKEY"]
            url = config["URL"]
            instance_id = config["INSTANCE_ID"]
            version = config["VERSION"]

            self["wml_credentials"] = {
                "username": username,
                "apikey": apikey,
                "url": url,
                "instance_id": instance_id,
                "version": version,
            }

            self["deployment_space_name"] = config["DEPLOYMENT_SPACE_NAME"]

        elif (
            "TOKEN" in config
            and "URL" in config
            and "INSTANCE_ID" in config
            and "VERSION" in config
            and "DEPLOYMENT_SPACE_NAME" in config
        ):
            token = config["TOKEN"]
            url = config["URL"]
            instance_id = config["INSTANCE_ID"]
            version = config["VERSION"]

            self["wml_credentials"] = {
                "token": token,
                "url": url,
                "instance_id": instance_id,
                "version": version,
            }

            self["deployment_space_name"] = config["DEPLOYMENT_SPACE_NAME"]

        else:
            raise MlflowException(
                "Missing environment variables", error_code=INVALID_PARAMETER_VALUE
            )

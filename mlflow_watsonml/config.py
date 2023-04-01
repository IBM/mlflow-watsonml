import json
import os

from dotenv import load_dotenv
from mlflow import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class Config(dict):
    def __init__(self):
        """
        Initializes constants from .env file and environment variables
        """
        super().__init__()

        load_dotenv()

        # Load the appropriate environment variables based on their presence
        if (
            "WML_CREDENTIALS_FILE" in os.environ
            and "DEPLOYMENT_SPACE_NAME" in os.environ
        ):
            wml_credentials_file = os.environ["WML_CREDENTIALS_FILE"]

            with open(wml_credentials_file, "r") as f:
                self["wml_credentials"] = json.load(f)

            self["deployment_space_name"] = os.environ["DEPLOYMENT_SPACE_NAME"]

        elif (
            "APIKEY" in os.environ
            and "LOCATION" in os.environ
            and "URL" in os.environ
            and "DEPLOYMENT_SPACE_NAME" in os.environ
        ):
            apikey = os.environ["APIKEY"]
            location = os.environ["LOCATION"]
            url = os.environ["URL"]

            self["wml_credentials"] = {
                "apikey": apikey,
                "url": url,
            }

            self["deployment_space_name"] = os.environ["DEPLOYMENT_SPACE_NAME"]

        elif (
            "TOKEN" in os.environ
            and "LOCATION" in os.environ
            and "URL" in os.environ
            and "DEPLOYMENT_SPACE_NAME" in os.environ
        ):
            token = os.environ["TOKEN"]
            location = os.environ["LOCATION"]
            url = os.environ["URL"]

            self["wml_credentials"] = {
                "token": token,
                "url": url,
            }
            self["deployment_space_name"] = os.environ["DEPLOYMENT_SPACE_NAME"]

        elif (
            "USERNAME" in os.environ
            and "PASSWORD" in os.environ
            and "URL" in os.environ
            and "INSTANCE_ID" in os.environ
            and "VERSION" in os.environ
            and "DEPLOYMENT_SPACE_NAME" in os.environ
        ):
            username = os.environ["USERNAME"]
            password = os.environ["PASSWORD"]
            url = os.environ["URL"]
            instance_id = os.environ["INSTANCE_ID"]
            version = os.environ["VERSION"]

            self["wml_credentials"] = {
                "username": username,
                "password": password,
                "url": url,
                "instance_id": instance_id,
                "version": version,
            }

            self["deployment_space_name"] = os.environ["DEPLOYMENT_SPACE_NAME"]

        elif (
            "USERNAME" in os.environ
            and "APIKEY" in os.environ
            and "URL" in os.environ
            and "INSTANCE_ID" in os.environ
            and "VERSION" in os.environ
            and "DEPLOYMENT_SPACE_NAME" in os.environ
        ):
            username = os.environ["USERNAME"]
            apikey = os.environ["APIKEY"]
            url = os.environ["URL"]
            instance_id = os.environ["INSTANCE_ID"]
            version = os.environ["VERSION"]

            self["wml_credentials"] = {
                "username": username,
                "apikey": apikey,
                "url": url,
                "instance_id": instance_id,
                "version": version,
            }

            self["deployment_space_name"] = os.environ["DEPLOYMENT_SPACE_NAME"]

        elif (
            "TOKEN" in os.environ
            and "URL" in os.environ
            and "INSTANCE_ID" in os.environ
            and "VERSION" in os.environ
            and "DEPLOYMENT_SPACE_NAME" in os.environ
        ):
            token = os.environ["TOKEN"]
            url = os.environ["URL"]
            instance_id = os.environ["INSTANCE_ID"]
            version = os.environ["VERSION"]

            self["wml_credentials"] = {
                "token": token,
                "url": url,
                "instance_id": instance_id,
                "version": version,
            }

            self["deployment_space_name"] = os.environ["DEPLOYMENT_SPACE_NAME"]

        else:
            raise MlflowException(
                "Missing environment variables", error_code=INVALID_PARAMETER_VALUE
            )

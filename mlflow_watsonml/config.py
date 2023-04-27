import json
import logging
import os
from typing import Dict, Optional

from dotenv import load_dotenv
from mlflow import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

LOGGER = logging.getLogger(__name__)

WML_CREDENTIALS = "wml_credentials"
DEPLOYMENT_SPACE_NAME = "deployment_space_name"
WML_CREDENTIALS_FILE = "wml_credentials_file"
APIKEY = "apikey"
LOCATION = "location"
URL = "url"
TOKEN = "token"
USERNAME = "username"
PASSWORD = "password"
INSTANCE_ID = "instance_id"
VERSION = "version"


class Config(dict):
    def __init__(self, config: Optional[Dict[str, str]] = None):
        """
        Initializes constants from input `config`, `.env` file or environment variables
        """
        super().__init__()

        if config is None:
            LOGGER.info(
                """Input credentials not provided. 
                Attempting to load credentials from environment variables."""
            )

            load_dotenv()
            config = {}

            if "DEPLOYMENT_SPACE_NAME" in os.environ.keys():
                config[DEPLOYMENT_SPACE_NAME] = os.getenv("DEPLOYMENT_SPACE_NAME")

            if "WML_CREDENTIALS_FILE" in os.environ.keys():
                config[WML_CREDENTIALS_FILE] = os.getenv("WML_CREDENTIALS_FILE")

            if "APIKEY" in os.environ.keys():
                config[APIKEY] = os.getenv("APIKEY")

            if "LOCATION" in os.environ.keys():
                config[LOCATION] = os.getenv("LOCATION")

            if "URL" in os.environ.keys():
                config[URL] = os.getenv("URL")

            if "TOKEN" in os.environ.keys():
                config[TOKEN] = os.getenv("TOKEN")

            if "USERNAME" in os.environ.keys():
                config[USERNAME] = os.getenv("USERNAME")

            if "PASSWORD" in os.environ.keys():
                config[PASSWORD] = os.getenv("PASSWORD")

            if "INSTANCE_ID" in os.environ.keys():
                config[INSTANCE_ID] = os.getenv("INSTANCE_ID")

            if "VERSION" in os.environ.keys():
                config[VERSION] = os.getenv("VERSION")

        self[DEPLOYMENT_SPACE_NAME] = config.get(DEPLOYMENT_SPACE_NAME, None)

        # Load the appropriate environment variables based on their presence
        if WML_CREDENTIALS_FILE in config:
            wml_credentials_file = config[WML_CREDENTIALS_FILE]

            with open(wml_credentials_file, "r") as f:
                self[WML_CREDENTIALS] = json.load(f)

        elif APIKEY in config and LOCATION in config and URL in config:
            apikey = config[APIKEY]
            location = config[LOCATION]
            url = config[URL]

            self[WML_CREDENTIALS] = {
                APIKEY: apikey,
                URL: url,
            }

        elif TOKEN in config and LOCATION in config and URL in config:
            token = config[TOKEN]
            location = config[LOCATION]
            url = config[URL]

            self[WML_CREDENTIALS] = {
                TOKEN: token,
                URL: url,
            }

        elif (
            USERNAME in config
            and PASSWORD in config
            and URL in config
            and INSTANCE_ID in config
            and VERSION in config
        ):
            username = config[USERNAME]
            password = config[PASSWORD]
            url = config[URL]
            instance_id = config[INSTANCE_ID]
            version = config[VERSION]

            self[WML_CREDENTIALS] = {
                USERNAME: username,
                PASSWORD: password,
                URL: url,
                INSTANCE_ID: instance_id,
                VERSION: version,
            }

        elif (
            USERNAME in config
            and APIKEY in config
            and URL in config
            and INSTANCE_ID in config
            and VERSION in config
        ):
            username = config[USERNAME]
            apikey = config[APIKEY]
            url = config[URL]
            instance_id = config[INSTANCE_ID]
            version = config[VERSION]

            self[WML_CREDENTIALS] = {
                USERNAME: username,
                APIKEY: apikey,
                URL: url,
                INSTANCE_ID: instance_id,
                VERSION: version,
            }

        elif (
            TOKEN in config
            and URL in config
            and INSTANCE_ID in config
            and VERSION in config
        ):
            token = config[TOKEN]
            url = config[URL]
            instance_id = config[INSTANCE_ID]
            version = config[VERSION]

            self[WML_CREDENTIALS] = {
                TOKEN: token,
                URL: url,
                INSTANCE_ID: instance_id,
                VERSION: version,
            }

        else:
            raise MlflowException(
                "Missing Credentials", error_code=INVALID_PARAMETER_VALUE
            )

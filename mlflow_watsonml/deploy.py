import logging
import os
import sys
import zipfile
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from ibm_watson_machine_learning.client import APIClient
from mlflow.deployments import BaseDeploymentClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, INVALID_PARAMETER_VALUE

from mlflow_watsonml.config import Config

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def target_help():
    # TODO: Improve
    help_string = (
        "The mlflow-watsonml plugin integrates IBM WatsonML "
        "with the MLFlow deployments API.\n\n"
        "Before using this plugin, you must set up a json file "
        "containing WML credentials and create an environment variable "
        "along with deployment space name. "
    )
    return help_string


def run_local(name, model_uri, flavor=None, config=None):
    # TODO: implement
    raise MlflowException("mlflow-watsonml does not currently support run_local.")


class WatsonMLDeploymentClient(BaseDeploymentClient):
    def __init__(self, target_uri):
        super().__init__(target_uri)

        self.wml_config = Config()
        self.connect(
            wml_credentials=self.wml_config["wml_credentials"],
            deployment_space_name=self.wml_config["deployment_space_name"],
        )

    def connect(self, wml_credentials: Dict, deployment_space_name: str) -> None:
        """Connect to WML APIClient and set the default deployment space

        Parameters
        ----------
        wml_credentials : Dict
            WML Credentials
        deployment_space_name : str
            Deployment space name

        Raises
        ------
        MlflowException
            _description_
        """
        try:
            self._wml_client = APIClient(wml_credentials=wml_credentials)
            space_uid = self._get_space_id_from_space_name(
                space_name=deployment_space_name
            )
            self._wml_client.set.default_space(space_uid=space_uid)

            LOGGER.info("Connected to WML Client successfully")

        except Exception as e:
            raise MlflowException(
                "Could not establish connection, check credentials and deployment space name."
                f"{e}",
                error_code=ENDPOINT_NOT_FOUND,
            )

    def create_deployment(
        self, name, model_uri, flavor=None, config=None, endpoint=None
    ):
        return super().create_deployment(name, model_uri, flavor, config, endpoint)

    def delete_deployment(self, name, config=None, endpoint=None):
        return super().delete_deployment(name, config, endpoint)

    def update_deployment(
        self, name, model_uri=None, flavor=None, config=None, endpoint=None
    ):
        return super().update_deployment(name, model_uri, flavor, config, endpoint)

    def list_deployments(self) -> List[Dict]:
        """_summary_

        Returns
        -------
        List[Dict]
            _description_
        """
        deployments = self.get_wml_client().deployments.get_details(get_all=True)[
            "resources"
        ]

        # `name` is a required key in each deployment
        for deployment in deployments:
            deployment["name"] = deployment["entity"]["name"]

        return deployments

    def list_models(self) -> List[Dict]:
        """_summary_

        Returns
        -------
        List[Dict]
            _description_
        """
        models = self.get_wml_client().repository.get_model_details(get_all=True)[
            "resources"
        ]

        for model in models:
            model["name"] = model["metadata"]["name"]

        return models

    def get_deployment(self, name: str) -> Dict:
        """_summary_

        Parameters
        ----------
        name : str
            deployment name

        Returns
        -------
        Dict
            deployment details

        Raises
        ------
        MlflowException
            _description_
        """
        deployments = self.list_deployments()

        try:
            return next(item for item in deployments if item["entity"]["name"] == name)

        except StopIteration as _:
            raise MlflowException(
                message=f"no deployment by the name {name} exists",
                error_code=ENDPOINT_NOT_FOUND,
            )

    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        return super().predict(deployment_name, inputs, endpoint)

    def get_wml_client(self) -> APIClient:
        """_summary_

        Returns
        -------
        APIClient
            _description_
        """
        return self._wml_client

    def _get_space_id_from_space_name(self, space_name: str) -> str:
        """Returns space ID from the space name

        Parameters
        ----------
        space_name : str
            space name

        Returns
        -------
        str
            space id

        Raises
        ------
        MlflowException
            _description_
        """
        spaces = self.get_wml_client().spaces.get_details()

        try:
            return next(
                item
                for item in spaces["resources"]
                if item["entity"]["name"] == space_name
            )["metadata"]["id"]
        except StopIteration as _:
            raise MlflowException(
                message=f"space {space_name} not found", error_code=ENDPOINT_NOT_FOUND
            )

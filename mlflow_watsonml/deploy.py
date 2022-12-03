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
from mlflow_watsonml.utils import *
from mlflow_watsonml.wml import *

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# CONSTANTS

_PYTHONMAJOR = sys.version_info.major
_PYTHONMINOR = sys.version_info.minor

if _PYTHONMAJOR == 3 and _PYTHONMINOR == 7:
    PYTHON_SPEC = f"default_py{_PYTHONMAJOR}.{_PYTHONMINOR}_opence"
else:
    PYTHON_SPEC = f"default_py{_PYTHONMAJOR}.{_PYTHONMINOR}"

PYTHON = "python"

DEFAULT_SOFTWARE_SPEC = "runtime-22.1-py3.9"
DEFAULT_MODEL_TYPE = "scikit-learn_1.0"


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
        self,
        name: str,
        model_uri: str,
        flavor: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> Dict:
        """_summary_

        Parameters
        ----------
        name : str
            _description_
        model_uri : str
            _description_
        flavor : Optional[str], optional
            _description_, by default None
        config : Optional[Dict], optional
            _description_, by default None

        Returns
        -------
        Dict
            _description_
        """
        if config is None:
            config = dict()

        client = self.get_wml_client()

        # check if a deployment by that name exists
        if self.deployment_exists(name) or self.model_exists(name):
            raise MlflowException(
                f"Deplyment {name} already exists. Use `update_deployment()` or use a different name",
                error_code=INVALID_PARAMETER_VALUE,
            )

        model_object = mlflow.sklearn.load_model(model_uri=model_uri)

        software_spec_type = config.get("software_spec_type", DEFAULT_SOFTWARE_SPEC)
        software_spec_uid = client.software_specifications.get_uid_by_name(
            software_spec_type
        )

        model_description = config.get("model_description", "no explanation")
        model_type = config.get("model_type", DEFAULT_MODEL_TYPE)

        model_details = store_model(
            client=client,
            model_object=model_object,
            software_spec_uid=software_spec_uid,
            name=name,
            model_description=model_description,
            model_type=model_type,
        )

        model_id = get_model_id_from_model_details(model_details=model_details)

        LOGGER.info("Stored Model Details = %s", model_details)
        LOGGER.info("Stored Model UID = %s", model_id)

        batch = config.get(batch, False)

        deployment_details = deploy_model(
            client=client,
            name=name,
            model_id=model_id,
            batch=batch,
        )

        return deployment_details

    def delete_deployment(self, name: str, config: Optional[Dict] = None) -> None:
        """_summary_

        Parameters
        ----------
        name : str
            _description_
        config : Optional[Dict], optional
            _description_, by default None

        Raises
        ------
        MlflowException
            _description_
        """
        client = self.get_wml_client()
        if self.deployment_exists(name) and self.model_exists(name):
            try:
                deployment_id = get_deployment_id_from_deployment_name(
                    client=client, name=name
                )
                client.deployments.delete(deployment_uid=deployment_id)

                model_id = get_model_id_from_model_name(client=client, name=name)
                client.repository.delete(model_id)

            except Exception as e:
                raise MlflowException(e)

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

    def deployment_exists(self, name: str) -> bool:
        """Checks if a deployment by the given name exists

        Parameters
        ----------
        name : str
            name of the deployment

        Returns
        -------
        bool
            True if the deployment exists else False
        """
        deployments = self.list_deployments()

        return any(item for item in deployments if item["name"] == name)

    def model_exists(self, name: str) -> bool:
        """Checks if a model by the given name exists

        Parameters
        ----------
        name : str
            name of the model

        Returns
        -------
        bool
            True if the model exists else False
        """
        models = self.list_models()

        return any(item for item in models if item["name"] == name)

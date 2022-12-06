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
        """
        try:
            self._wml_client = APIClient(wml_credentials=wml_credentials)
            space_uid = get_space_id_from_space_name(
                client=self._wml_client,
                space_name=deployment_space_name,
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
        """Deploy the model at `model_uri` to a WML target

        Parameters
        ----------
        name : str
            name of the deployment
        model_uri : str
            URI (local or remote) of the model
        flavor : Optional[str], optional
            flavor of the deployed model, by default None
        config : Optional[Dict], optional
            configuration parameters, by default None

        Returns
        -------
        Dict
            deployment details dictionary
        """
        if config is None:
            config = dict()

        client = self.get_wml_client()

        # check if a deployment by that name exists
        if deployment_exists(client=client, name=name) or model_exists(
            client=client, name=name
        ):
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

        model_id = get_model_id_from_model_details(
            client=client, model_details=model_details
        )

        LOGGER.info("Stored Model Details = %s", model_details)
        LOGGER.info("Stored Model UID = %s", model_id)

        batch = config.get("batch", False)

        deployment_details = deploy_model(
            client=client,
            name=name,
            model_id=model_id,
            batch=batch,
        )

        return deployment_details

    def delete_deployment(self, name: str, config: Optional[Dict] = None) -> None:
        """Delete the deployment at the provided `name` from WML

        Parameters
        ----------
        name : str
            name of the deployment
        config : Optional[Dict], optional
            configuration parameters, by default None
        """
        client = self.get_wml_client()
        if deployment_exists(client=client, name=name) or model_exists(
            client=client, name=name
        ):
            delete_model(client=client, name=name)

    def update_deployment(
        self, name, model_uri=None, flavor=None, config=None, endpoint=None
    ):
        return super().update_deployment(name, model_uri, flavor, config, endpoint)

    def predict(self, deployment_name=None, df=Optional[pd.DataFrame]) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        deployment_name : _type_, optional
            _description_, by default None
        df : _type_, optional
            _description_, by default Optional[pd.DataFrame]

        Returns
        -------
        pd.DataFrame
            _description_

        Raises
        ------
        MlflowException
            _description_
        """
        if self._wml_client is None:
            raise MlflowException(
                message="Deployment doesn't exist. Call `create_deployment()`",
                error_code=ENDPOINT_NOT_FOUND,
            )

        client = self.get_wml_client()

        scoring_payload = {
            self._wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{"values": df}]
        }

        deployment_id = get_deployment_id_from_deployment_name(
            client=client,
            deployment_name=deployment_name,
        )

        predictions = client.deployments.score(
            deployment_id=deployment_id, meta_props=scoring_payload
        )["predictions"]

        fields = predictions[0]["fields"]

        frames = []
        for prediction in predictions:
            frames.extend(prediction["values"])

        ans = pd.DataFrame(frames, columns=fields)

        return ans

    def list_deployments(self) -> List[Dict]:
        return list_deployments(client=self.get_wml_client())

    def get_deployment(self, name: str) -> Dict:
        return get_deployment(client=self.get_wml_client(), name=name)

    def get_wml_client(self) -> APIClient:
        """_summary_

        Returns
        -------
        APIClient
            _description_
        """
        if hasattr(self, "_wml_client"):
            return self._wml_client
        else:
            raise MlflowException(f"No WML Client found")

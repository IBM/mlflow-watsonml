import logging
import sys
from typing import Dict, List, Optional

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
DEFAULT_SOFTWARE_SPEC = "runtime-22.2-py3.10"
DEFAULT_MODEL_TYPE = "scikit-learn_1.1"


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
    def __init__(self, target_uri: str = "watsonml", config: Optional[Dict] = None):
        """Initialize a WML `APIClient`. The method has an optional parameter called
        `config` which should have the WML credentials. If `config` is `None`, then
        the plugin will try to search for WML credentials in `.env` file or the
        environment variables.

        Parameters
        ----------
        target_uri : str, optional
            target uri for mlflow deployment, by default "watsonml"
        config : Optional[Dict], optional
            wml credentials, by default None
        """
        super().__init__(target_uri)

        self.wml_config = Config(config=config)
        self.connect(
            wml_credentials=self.wml_config["wml_credentials"],
            endpoint=self.wml_config["deployment_space_name"],
        )

    def connect(self, wml_credentials: Dict, endpoint: Optional[str] = None) -> None:
        """Connect to WML APIClient and set the default deployment space

        Parameters
        ----------
        wml_credentials : Dict
            WML Credentials
        endpoint : Optional[str]
            Deployment space name
        """
        try:
            client = APIClient(wml_credentials=wml_credentials)
            self.endpoint = endpoint
            LOGGER.info("Connected to WML Client successfully")

        except Exception as e:
            raise MlflowException(
                "Could not establish connection, check wml credentials." f"{e}",
                error_code=ENDPOINT_NOT_FOUND,
            )

        self._wml_client = client

    def get_wml_client(self, endpoint: Optional[str] = None) -> APIClient:
        """Returns WML API client

        Parameters
        ----------
        endpoint : Optional[str], optional
            deployment space name, by default None

        Returns
        -------
        APIClient
            WML client
        """
        client = self._wml_client

        if endpoint is not None:
            try:
                client = set_deployment_space(
                    client=client, deployment_space_name=endpoint
                )
            except Exception as e:
                raise MlflowException(e)

            return client

        elif self.endpoint is not None:
            try:
                client = set_deployment_space(
                    client=client, deployment_space_name=self.endpoint
                )
            except Exception as e:
                raise MlflowException(e)

            return client

        else:
            raise MlflowException(
                f"Deployment space name not set. Please set it using `endpoint` parameter"
            )

    def create_deployment(
        self,
        name: str,
        model_uri: str,
        flavor: Optional[str] = None,
        config: Optional[Dict] = None,
        endpoint: Optional[str] = None,
    ) -> Dict:
        """Deploy a model at `model_uri` to a WML target. this method blocks until
        deployment completes (i.e. until it's possible to perform inference with the deployment).
        In the case of conflicts (e.g. if it's not possible to create the specified deployment
        without due to conflict with an existing deployment), raises a
        :py:class:`mlflow.exceptions.MlflowException`.

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
        endpoint : Optional[str], optional
            deployment space name, by default None

        Returns
        -------
        Dict
            deployment details dictionary
        """
        if config is None:
            config = dict()

        client = self.get_wml_client(endpoint=endpoint)

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
        software_spec_uid = client.software_specifications.get_id_by_name(
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

    def update_deployment(
        self,
        name: str,
        model_uri: Optional[str] = None,
        flavor: Optional[str] = None,
        config: Optional[Dict] = None,
        endpoint: Optional[str] = None,
    ) -> Dict:
        """Update the deployment with the specified name. You can update the URI of the model, the
        flavor of the deployed model (in which case the model URI must also be specified), and/or
        any WML-specific attributes of the deployment (via `config`). By default, this method
        blocks until deployment completes (i.e. until it's possible to perform inference
        with the updated deployment).

        Parameters
        ----------
        name : str
            Unique name of the deployment to update
        model_uri : Optional[str], optional
            URI of a new model to deploy, by default None
        flavor : Optional[str], optional
            new model flavor to use for deployment. If provided,
            ``model_uri`` must also be specified. If ``flavor`` is unspecified but
            ``model_uri`` is specified, a default flavor will be chosen and the
            deployment will be updated using that flavor., by default None
        config : Optional[Dict], optional
            dict containing updated WML-specific configuration for the
            deployment, by default None
        endpoint : Optional[str], optional
            deployment space name, by default None

        Returns
        -------
        Dict
            deployment details dictionary
        """
        self.delete_deployment(name=name, config=config, endpoint=endpoint)

        deployment_details = self.create_deployment(
            name=name,
            model_uri=model_uri,
            flavor=flavor,
            config=config,
            endpoint=endpoint,
        )

        return deployment_details

    def delete_deployment(
        self, name: str, config: Optional[Dict] = None, endpoint: Optional[str] = None
    ):
        """Delete the deployment with name ``name`` from WML. Deletion is idempotent
        (i.e. deletion does not fail if retried on a non-existent deployment).

        Parameters
        ----------
        name : str
            name of the deployment to delete
        config : Optional[Dict], optional
            configuration parameters, by default None
        endpoint : Optional[str], optional
            deployment space name, by default None
        """
        if config is None:
            config = dict()

        client = self.get_wml_client(endpoint=endpoint)

        if deployment_exists(client=client, name=name):
            delete_deployment(client=client, name=name)

        if model_exists(client=client, name=name):
            delete_model(client=client, name=name)

    def list_deployments(self, endpoint: Optional[str] = None):
        """List deployments. This method returns an unpaginated list of all deployments

        Parameters
        ----------
        endpoint : Optional[str], optional
            deployment space name, by default None

        Returns
        -------
        List[Dict]
            A list of dicts corresponding to deployments. Each dict is guaranteed to
            contain a 'name' key containing the deployment name. The other fields of
            the returned dictionary and their types follow WML deployment details convention.
        """
        return list_deployments(client=self.get_wml_client(endpoint=endpoint))

    def get_deployment(self, name: str, endpoint: Optional[str] = None):
        """Returns a dictionary describing the specified deployment, throwing a
        :py:class:`mlflow.exceptions.MlflowException` if no deployment exists with the provided
        name.
        The dict is guaranteed to contain a 'name' key containing the deployment name.
        The other fields of the returned dictionary and their types follow WML convention.

        Parameters
        ----------
        name : str
            name of the deployment to fetch
        endpoint : Optional[str], optional
            deployment space name, by default None

        Returns
        -------
        Dict
            A dict corresponding to the retrieved deployment. The dict is guaranteed to
            contain a 'name' key corresponding to the deployment name.
        """
        return get_deployment(client=self.get_wml_client(endpoint=endpoint), name=name)

    def predict(
        self,
        deployment_name: str,
        inputs: Optional[pd.DataFrame] = None,
        endpoint: Optional[str] = None,
    ):
        """Compute predictions on inputs using the specified deployment

        Parameters
        ----------
        deployment_name : str
            Name of deployment to predict against
        inputs : pd.DataFrame, optional
            Input data (or arguments) to pass to the deployment for inference,
            by default Optional[pd.DataFrame]
        endpoint : Optional[str], optional
            deployment space name, by default None

        Returns
        -------
        pd.DataFrame
            Model predictions as pandas.DataFrame
        """
        client = self.get_wml_client(endpoint=endpoint)

        scoring_payload = {
            self._wml_client.deployments.ScoringMetaNames.INPUT_DATA: [
                {"values": inputs}
            ]
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

    def explain(self, deployment_name=None, df=None, endpoint=None):
        raise NotImplementedError()

    def create_endpoint(self, name, config=None):
        raise NotImplementedError()

    def update_endpoint(self, endpoint, config=None):
        raise NotImplementedError()

    def delete_endpoint(self, endpoint):
        raise NotImplementedError()

    def list_endpoints(self):
        return list_endpoints(client=self.get_wml_client())

    def get_endpoint(self, endpoint):
        return get_endpoint(client=self.get_wml_client(), endpoint=endpoint)

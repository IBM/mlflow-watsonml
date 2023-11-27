from typing import Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
from ibm_watson_machine_learning.client import APIClient
from mlflow.deployments import BaseDeploymentClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, INVALID_PARAMETER_VALUE

from mlflow_watsonml.config import Config
from mlflow_watsonml.logging import LOGGER
from mlflow_watsonml.utils import *
from mlflow_watsonml.wml import *


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
    raise MlflowException("mlflow-watsonml does not currently support run_local.")


class WatsonMLDeploymentClient(BaseDeploymentClient):
    def __init__(self, target_uri: str = "watsonml", config: Optional[Dict] = None):
        """Initialize a WML `APIClient`. The method has an optional parameter called
        `config` which should have the WML credentials. If `config` is `None`, then
        the plugin will try to search for WML credentials in `.env` file or the
        environment variables.

        Refer to the following links for setting up the credentials -

        1. [Cloud Pak for Data as a Service](https://ibm.github.io/watson-machine-learning-sdk/setup_cloud.html#authentication)
        2. [Cloud Pak for Data](https://ibm.github.io/watson-machine-learning-sdk/setup_cpd.html#authentication)

        Parameters
        ----------
        target_uri : str, optional
            Target URI for mlflow deployment, by default "watsonml"
        config : Optional[Dict], optional
            WML Credentials, by default None
        """
        super().__init__(target_uri)

        self.wml_config = Config(config=config)
        self.connect(wml_credentials=self.wml_config["wml_credentials"])

    def connect(self, wml_credentials: Dict) -> None:
        """Connect to WML APIClient and set the default deployment space

        Parameters
        ----------
        wml_credentials : Dict
            WML Credentials
        """
        try:
            client = APIClient(wml_credentials=wml_credentials)
            LOGGER.info("Connected to WML Client successfully")

        except Exception as e:
            LOGGER.exception(e)
            raise MlflowException(
                "Could not establish connection, check WML credentials." f"{e}",
                error_code=ENDPOINT_NOT_FOUND,
            )

        self._wml_client = client

    def get_wml_client(self, endpoint: str) -> APIClient:
        """Returns WML API client

        Parameters
        ----------
        endpoint : str
            deployment space name

        Returns
        -------
        APIClient
            WML client
        """
        client = self._wml_client

        try:
            space_uid = get_space_id_from_space_name(
                client=client,
                space_name=endpoint,
            )

            if space_uid is None:
                raise MlflowException(
                    f"Endpoint {endpoint} not found.",
                    error_code=ENDPOINT_NOT_FOUND,
                )
            client.set.default_space(space_uid=space_uid)

            LOGGER.info(
                f"Set deployment space to {endpoint} with space id - {space_uid}"
            )

        except Exception as e:
            LOGGER.exception(e)
            raise MlflowException(f"Failed to set deployment space {endpoint}", f"{e}")

        return client

    def create_deployment(
        self,
        name: str,
        model_uri: str,
        flavor: str,
        config: Optional[Dict],
        endpoint: str,
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
        flavor : str
            flavor of the deployed model
        config : Dict
            configuration parameters for wml deployment.
            possible optional configuration keys are -
            - "software_spec_name" : name of the software specification to reuse
            - "conda_yaml" : filepath of conda.yaml file
            - "custom_packages": a list of str - zip file paths of the packages
            - "rewrite_software_spec": bool whether to rewrite the software spec
            - "hardware_spec_name" : name of the hardware specification to use (Default: XS)
        endpoint : str
            deployment space name

        Returns
        -------
        Dict
            deployment details dictionary
        """
        client = self.get_wml_client(endpoint=endpoint)

        if config is None:
            config = dict()

        # check if a deployment by that name exists
        if deployment_exists(client=client, name=name):
            raise MlflowException(
                f"Deployment {name} already exists. Use `update_deployment()` or use a different name",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if "software_spec_name" in config.keys():
            software_spec_id = client.software_specifications.get_id_by_name(
                config["software_spec_name"]
            )

            if software_spec_id == "Not Found":
                raise MlflowException(
                    f"Software Specification {config['software_spec_name']} not found.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        else:
            if "conda_yaml" in config.keys():
                conda_yaml = config["conda_yaml"]
            else:
                conda_yaml = mlflow.pyfunc.get_model_dependencies(
                    model_uri=model_uri, format="conda"
                )  # other option is to have a default conda_yaml for each flavor

            custom_packages: List[str] = config.get("custom_packages")
            rewrite: bool = config.get("rewrite_software_spec", False)

            software_spec_id = create_custom_software_spec(
                client=client,
                name=f"{name}_sw_spec",
                custom_packages=custom_packages,
                conda_yaml=conda_yaml,
                rewrite=rewrite,
            )

        artifact_name = f"{name}_v1"
        environment_variables = get_mlflow_config()

        artifact_id, revision_id = store_or_update_artifact(
            client=client,
            model_uri=model_uri,
            artifact_name=artifact_name,
            flavor=flavor,
            software_spec_id=software_spec_id,
            environment_variables=environment_variables,
        )

        batch = config.get("batch", False)

        hardware_spec_name = config.get("hardware_spec_name", "XS")
        if hardware_spec_name is not None:
            hardware_spec_id = client.hardware_specifications.get_id_by_name(
                hardware_spec_name
            )
            if hardware_spec_id == "Not Found":
                LOGGER.warn(
                    f"Hardware Specification - {hardware_spec_name} not found. Using default."
                )
                hardware_spec_id = None
        else:
            hardware_spec_id = None

        deployment_details = deploy(
            client=client,
            name=name,
            artifact_id=artifact_id,
            revision_id=revision_id,
            batch=batch,
            environment_variables=environment_variables,
            hardware_spec_id=hardware_spec_id,
        )

        return deployment_details

    def update_deployment(
        self,
        name: str,
        model_uri: str,
        flavor: str,
        config: Optional[Dict],
        endpoint: str,
    ) -> Dict:
        """Update the deployment with the specified name. You can update the URI of the model, the
        flavor of the deployed model (in which case the model URI must also be specified). By default,
        this method blocks until deployment completes (i.e. until it's possible to perform inference
        with the updated deployment).

        Parameters
        ----------
        name : str
            Unique name of the deployment to update
        model_uri : str
            URI of a new model to deploy
        flavor : str
            new model flavor to use for deployment. If provided,
            ``model_uri`` must also be specified.
        config : Optional[Dict], optional
            dict containing updated WML-specific configuration for the
        endpoint : str
            deployment space name

        Returns
        -------
        Dict
            deployment details dictionary
        """
        client = self.get_wml_client(endpoint=endpoint)

        if config is None:
            config = dict()

        # check if a deployment by that name exists
        if not deployment_exists(client=client, name=name):
            raise MlflowException(
                f"Deployment {name} doesn't exist. Use `create_deployment()`",
                error_code=INVALID_PARAMETER_VALUE,
            )

        current_deployment = self.get_deployment(name=name, endpoint=endpoint)
        artifact_id = current_deployment["entity"]["asset"]["id"]
        artifact_rev = int(current_deployment["entity"]["asset"]["rev"])

        new_artifact_name = f"{name}_v{artifact_rev+1}"

        if "software_spec_name" in config.keys():
            software_spec_id = client.software_specifications.get_id_by_name(
                config["software_spec_name"]
            )

            if software_spec_id == "Not Found":
                raise MlflowException(
                    f"Software Specification {config['software_spec_name']} not found.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        else:
            if "conda_yaml" in config.keys():
                conda_yaml = config["conda_yaml"]
            else:
                conda_yaml = mlflow.pyfunc.get_model_dependencies(
                    model_uri=model_uri, format="conda"
                )  # other option is to have a default conda_yaml for each flavor

            custom_packages: List[str] = config.get("custom_packages")

            software_spec_id = create_custom_software_spec(
                client=client,
                name=f"{name}_sw_spec",
                custom_packages=custom_packages,
                conda_yaml=conda_yaml,
                rewrite=True,
            )
        environment_variables = get_mlflow_config()
        artifact_id, revision_id = store_or_update_artifact(
            client=client,
            model_uri=model_uri,
            artifact_name=new_artifact_name,
            flavor=flavor,
            software_spec_id=software_spec_id,
            environment_variables=environment_variables,
        )

        deployment_details = update_deployment(
            client=client,
            name=name,
            artifact_id=artifact_id,
            revision_id=revision_id,
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

    def list_deployments(self, endpoint: str):
        """List deployments. This method returns an unpaginated list of all deployments

        Parameters
        ----------
        endpoint : str
            deployment space name

        Returns
        -------
        List[Dict]
            A list of dicts corresponding to deployments. Each dict is guaranteed to
            contain a 'name' key containing the deployment name. The other fields of
            the returned dictionary and their types follow WML deployment details convention.
        """
        return list_deployments(client=self.get_wml_client(endpoint=endpoint))

    def get_deployment(self, name: str, endpoint: str):
        """Returns a dictionary describing the specified deployment, throwing a
        :py:class:`mlflow.exceptions.MlflowException` if no deployment exists with the provided
        name.
        The dict is guaranteed to contain a 'name' key containing the deployment name.
        The other fields of the returned dictionary and their types follow WML convention.

        Parameters
        ----------
        name : str
            name of the deployment to fetch
        endpoint : str
            deployment space name

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
        inputs: Union[pd.DataFrame, np.ndarray, List[Any], Dict[str, Any]],
        endpoint: str,
    ) -> Union[np.ndarray, pd.DataFrame, pd.Series, List]:
        """Compute predictions on inputs using the specified deployment

        predict(
            model_input:
        ) ->

        Parameters
        ----------
        deployment_name : str
            Name of deployment to predict against
        inputs : pd.DataFrame
            Input data (or arguments) to pass to the deployment for inference,
        endpoint : str
            deployment space name

        Returns
        -------
        pd.DataFrame
            Model predictions as pandas.DataFrame
        """
        client = self.get_wml_client(endpoint=endpoint)

        deployment_details = self.get_deployment(
            name=deployment_name, endpoint=endpoint
        )

        scoring_payload = {
            client.deployments.ScoringMetaNames.INPUT_DATA: [{"values": inputs}]
        }

        if "custom" in deployment_details["entity"].keys():
            scoring_payload[
                client.deployments.ScoringMetaNames.ENVIRONMENT_VARIABLES
            ] = deployment_details["entity"]["custom"]

        deployment_id = client.deployments.get_id(deployment_details=deployment_details)

        predictions = client.deployments.score(
            deployment_id=deployment_id, meta_props=scoring_payload
        )["predictions"]

        if (
            len(predictions) > 0
            and isinstance(predictions[0], dict)
            and "fields" in predictions[0].keys()
        ):
            fields = predictions[0]["fields"]

            frames = []
            for prediction in predictions:
                frames.extend(prediction["values"])

            ans = pd.DataFrame(frames, columns=fields)
            return ans

        return predictions  # ans

    def explain(self, deployment_name=None, df=None, endpoint=None):
        raise NotImplementedError()

    def create_endpoint(self, name: str, config: Optional[Dict] = None) -> Dict:
        """
        Create an endpoint with the specified target. By default, this method should block until
        creation completes (i.e. until it's possible to create a deployment within the endpoint).
        In the case of conflicts (e.g. if it's not possible to create the specified endpoint
        due to conflict with an existing endpoint), raises a
        :py:class:`mlflow.exceptions.MlflowException`. See target-specific plugin documentation
        for additional detail on support for asynchronous creation and other configuration.

        :param name: Unique name to use for endpoint. If another endpoint exists with the same
                     name, raises a :py:class:`mlflow.exceptions.MlflowException`.
        :param config: (optional) Dict containing target-specific configuration for the
                       endpoint.
        :return: Dict corresponding to created endpoint, which must contain the 'name' key.
        """
        client = self._wml_client

        if config is None:
            config = dict()

        metadata = dict()
        metadata[client.spaces.ConfigurationMetaNames.NAME] = name

        if client.spaces.ConfigurationMetaNames.DESCRIPTION in config.keys():
            metadata[client.spaces.ConfigurationMetaNames.DESCRIPTION] = config[
                client.spaces.ConfigurationMetaNames.DESCRIPTION
            ]

        endpoint_details = client.spaces.store(
            meta_props=metadata, background_mode=False
        )

        return endpoint_details

    def update_endpoint(self, endpoint, config=None):
        """
        Update the endpoint with the specified name. You can update any target-specific attributes
        of the endpoint (via `config`). By default, this method should block until the update
        completes (i.e. until it's possible to create a deployment within the endpoint). See
        target-specific plugin documentation for additional detail on support for asynchronous
        update and other configuration.

        :param endpoint: Unique name of endpoint to update
        :param config: (optional) dict containing target-specific configuration for the
                       endpoint
        :return: None
        """
        raise NotImplementedError()

    def delete_endpoint(self, endpoint):
        """
        Delete the endpoint from the specified target. Deletion should be idempotent (i.e. deletion
        should not fail if retried on a non-existent deployment).

        :param endpoint: Name of endpoint to delete
        :return: None
        """
        client = self._wml_client

        endpoint_id = get_space_id_from_space_name(client=client, space_name=endpoint)

        if endpoint_id is not None:
            client.spaces.delete(space_id=endpoint_id)

    def list_endpoints(self):
        """
        List endpoints in the specified target. This method is expected to return an
        unpaginated list of all endpoints (an alternative would be to return a dict with
        an 'endpoints' field containing the actual endpoints, with plugins able to specify
        other fields, e.g. a next_page_token field, in the returned dictionary for pagination,
        and to accept a `pagination_args` argument to this method for passing
        pagination-related args).

        :return: A list of dicts corresponding to endpoints. Each dict is guaranteed to
                 contain a 'name' key containing the endpoint name. The other fields of
                 the returned dictionary and their types may vary across targets.
        """
        client = self._wml_client
        endpoints = client.spaces.get_details(get_all=True)["resources"]

        for endpoint in endpoints:
            endpoint["name"] = endpoint["entity"]["name"]

        return endpoints

    def get_endpoint(self, endpoint):
        client = self._wml_client
        deployment_space_id = get_space_id_from_space_name(
            client=client, space_name=endpoint
        )
        endpoint_details = client.spaces.get_details(space_id=deployment_space_id)
        return endpoint_details

    def create_custom_wml_spec(
        self,
        name: str,
        custom_packages: Optional[List[str]],
        conda_yaml: Optional[str],
        endpoint: str,
        rewrite: bool = False,
    ) -> str:
        """Create a custom WML Software Specification

        Parameters
        ----------
        name : str
            name for the software specification
        custom_packages : Optional[List[str]]
            a list of zip file paths for custom packages
        conda_yaml : Optional[str]
            file path to conda.yaml file
        endpoint : str
            deployment space name
        rewrite : bool, optional
            whether to rewrite the existing software specification, by default False

        Returns
        -------
        str
            id of the software specification
        """
        client = self.get_wml_client(endpoint=endpoint)
        software_spec_id = create_custom_software_spec(
            client=client,
            name=name,
            custom_packages=custom_packages,
            conda_yaml=conda_yaml,
            rewrite=rewrite,
        )

        return software_spec_id

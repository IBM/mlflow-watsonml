import logging
import os
import zipfile
from typing import Dict, List, Optional

import yaml
from ibm_watson_machine_learning.client import APIClient
from mlflow.exceptions import ENDPOINT_NOT_FOUND, MlflowException

LOGGER = logging.getLogger(__name__)


def list_models(client: APIClient) -> List[Dict]:
    """lists models in WML repository

    Parameters
    ----------
    client : APIClient
        WML client

    Returns
    -------
    List[Dict]
        list of model details dictionary
    """
    models = client.repository.get_model_details(get_all=True)["resources"]

    for model in models:
        model["name"] = model["metadata"]["name"]

    return models


def get_model_id_from_model_name(client: APIClient, model_name: str) -> str:
    """Returns model ID from model name

    Parameters
    ----------
    model_name : str
        model name

    Returns
    -------
    str
        model id
    """
    models = list_models(client=client)

    try:
        return next(item for item in models if item["name"] == model_name)["metadata"][
            "id"
        ]
    except StopIteration as _:
        message = f"model {model_name} not found"
        LOGGER.exception(message)
        raise MlflowException(message=message, error_code=ENDPOINT_NOT_FOUND)


def model_exists(client: APIClient, name: str) -> bool:
    """Checks if a model by the given name exists

    Parameters
    ----------
    client : APIClient
        WML client
    name : str
        name of the model

    Returns
    -------
    bool
        True if the model exists else False
    """
    models = list_models(client=client)

    return any(item for item in models if item["name"] == name)


def list_deployments(client: APIClient) -> List[Dict]:
    """lists WML deployments

    Parameters
    ----------
    client : APIClient
        WML client

    Returns
    -------
    List[Dict]
        list of deployment details dictionary
    """
    deployments = client.deployments.get_details(get_all=True)["resources"]

    # `name` is a required key in each deployment
    for deployment in deployments:
        deployment["name"] = deployment["metadata"]["name"]

    return deployments


def get_deployment(client: APIClient, name: str) -> Dict:
    """retreive deployment details

    Parameters
    ----------
    client : APIClient
        WML client
    name : str
        name of the deployment

    Returns
    -------
    Dict
        deployment details dictionary
    """
    deployments = list_deployments(client=client)

    try:
        return next(item for item in deployments if item["name"] == name)

    except StopIteration as _:
        message = f"no deployment by the name {name} exists"
        LOGGER.exception(message)
        raise MlflowException(
            message=message,
            error_code=ENDPOINT_NOT_FOUND,
        )


def get_deployment_id_from_deployment_name(
    client: APIClient, deployment_name: str
) -> str:
    """Returns deployment ID from deployment name

    Parameters
    ----------
    deployment_name : str
        deployment name

    Returns
    -------
    str
        deployment id
    """
    return client.deployments.get_id(
        get_deployment(client=client, name=deployment_name)
    )


def deployment_exists(client: APIClient, name: str) -> bool:
    """Checks if a deployment by the given name exists

    Parameters
    ----------
    client : APIClient
        WML client
    name : str
        name of the deployment

    Returns
    -------
    bool
        True if the deployment exists else False
    """
    deployments = list_deployments(client=client)
    return any(item for item in deployments if item["name"] == name)


def get_space_id_from_space_name(client: APIClient, space_name: str) -> Optional[str]:
    """Returns space ID from the space name

    Parameters
    ----------
    client : APIClient
        WML client
    space_name : str
        space name

    Returns
    -------
    str | None
        space id
    """
    spaces = client.spaces.get_details(get_all=True)["resources"]

    try:
        return next(item for item in spaces if item["entity"]["name"] == space_name)[
            "metadata"
        ]["id"]
    except StopIteration as _:
        message = f"space {space_name} not found"
        LOGGER.warn(message)
        # raise MlflowException(message=message, error_code=ENDPOINT_NOT_FOUND)
        return None


def software_spec_exists(client: APIClient, name: str) -> bool:
    """Check if a given software specification exists.

    Parameters
    ----------
    client : APIClient
        WML client
    name : str
        name of the software specification

    Returns
    -------
    bool
        True if the software specification exists, else False
    """
    software_spec = client.software_specifications.get_id_by_name(sw_spec_name=name)
    return software_spec != "Not Found"


def get_software_spec_from_deployment_name(
    client: APIClient, deployment_name: str
) -> str:
    """Get software specification id for the given deployment

    Parameters
    ----------
    client : APIClient
        WML client
    deployment_name : str
        name of the deployment

    Returns
    -------
    str
        software specification id
    """
    deployment = get_deployment(client=client, name=deployment_name)
    model_id = deployment["entity"]["asset"]["id"]
    software_spec_id = client.repository.get_model_details(model_uid=model_id)[
        "entity"
    ]["software_spec"]["id"]

    return software_spec_id


def is_zipfile(file_path: str) -> bool:
    """Utility method to check if the given file path is a valid zip file.

    Parameters
    ----------
    file_path : str
        path to zip file

    Returns
    -------
    bool
        True if it is a valid zip file, else False
    """
    if not os.path.isfile(file_path):
        return False

    try:
        with zipfile.ZipFile(file_path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def refine_conda_yaml(conda_yaml: str) -> str:
    with open(conda_yaml, "r", encoding="utf-8") as f:
        env_data = yaml.safe_load(f)

    pip_dependencies = []

    for dep in env_data["dependencies"]:
        if isinstance(dep, dict):
            if "pip" in dep.keys():
                pip_dependencies.extend(dep["pip"])

    refined_env = {
        "channels": ["defaults"],
        "dependencies": [
            "pip",
            {"pip": pip_dependencies},
        ],
        "name": "mlflow-env",
    }

    with open(conda_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(refined_env, f)

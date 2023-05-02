from typing import Dict, List

from ibm_watson_machine_learning.client import APIClient
from mlflow.exceptions import ENDPOINT_NOT_FOUND, MlflowException
from tabulate import tabulate


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
        deployment["name"] = deployment["entity"]["name"]

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
        return next(item for item in deployments if item["entity"]["name"] == name)

    except StopIteration as _:
        raise MlflowException(
            message=f"no deployment by the name {name} exists",
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
    return get_deployment(client=client, name=deployment_name)["metadata"]["id"]


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
        raise MlflowException(
            message=f"model {model_name} not found", error_code=ENDPOINT_NOT_FOUND
        )


def get_space_id_from_space_name(client: APIClient, space_name: str) -> str:
    """Returns space ID from the space name

    Parameters
    ----------
    client : APIClient
        WML client
    space_name : str
        space name

    Returns
    -------
    str
        space id
    """
    spaces = client.spaces.get_details()["resources"]

    try:
        return next(item for item in spaces if item["entity"]["name"] == space_name)[
            "metadata"
        ]["id"]
    except StopIteration as _:
        raise MlflowException(
            message=f"space {space_name} not found", error_code=ENDPOINT_NOT_FOUND
        )


def get_model_id_from_model_details(client: APIClient, model_details: Dict) -> str:
    """Return model ID from model details dictionary

    Parameters
    ----------
    model_details : Dict
        model details dictionary

    Returns
    -------
    str
        model id
    """
    model_id = client.repository.get_model_id(model_details=model_details)
    return model_id


def get_deployment_id_from_deployment_details(
    client: APIClient, deployment_details: Dict
) -> str:
    """Return deployment ID from deployment details dictionary

    Parameters
    ----------
    client : APIClient
        WML client
    deployment_details : Dict
        deployment details dictionary

    Returns
    -------
    str
        deployment id
    """
    deployment_id = client.deployments.get_id(deployment_details=deployment_details)
    return deployment_id


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


def print_package_specifications(
    client: APIClient, software_spec: str = "runtime-22.2-py3.10"
) -> None:
    """Print the package spec along with version to use as reference for creating an env

    Parameters
    ----------
    client : APIClient
        WML Client
    software_spec : str, optional
        Software Spec, by default "runtime-22.2-py3.10"
    """
    pkg_specs = client.software_specifications.get_details(
        client.software_specifications.get_id_by_name(software_spec)
    )["entity"]["software_specification"]["software_configuration"]["included_packages"]

    print(tabulate(pkg_specs, headers="keys"))


def list_endpoints(client: APIClient) -> List[Dict]:
    endpoints = client.spaces.get_details(get_all=True)["resources"]
    return endpoints


def get_endpoint(client: APIClient, endpoint: str):
    deployment_space_id = get_space_id_from_space_name(
        client=client, space_name=endpoint
    )
    endpoint_details = client.spaces.get_details(space_id=deployment_space_id)
    return endpoint_details


def list_software_specs(client: APIClient) -> List[Dict]:
    sw_specs = client.software_specifications.get_details()["resources"]
    return sw_specs


def get_software_spec(client: APIClient, sw_spec: str) -> Dict:
    sw_specs = list_software_specs(client=client)

    try:
        return next(item for item in sw_specs if item["metadata"]["name"] == sw_spec)[
            "metadata"
        ]["id"]
    except StopIteration as _:
        raise MlflowException(
            message=f"Software Specifiction - {sw_spec} not found",
            error_code=ENDPOINT_NOT_FOUND,
        )


def software_spec_exists(client: APIClient, sw_spec: str) -> bool:
    sw_specs = list_software_specs(client=client)

    return any(item for item in sw_specs if item["metadata"]["name"] == sw_spec)

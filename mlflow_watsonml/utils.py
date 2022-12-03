from typing import Any, Dict

from ibm_watson_machine_learning.client import APIClient
from mlflow.exceptions import MlflowException


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
    return self.get_deployment(name=deployment_name)["metadata"]["id"]


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

    Raises
    ------
    MlflowException
        _description_
    """
    models = self.list_models()

    try:
        return next(item for item in models if item["name"] == model_name)["metadata"][
            "id"
        ]
    except StopIteration as _:
        raise MlflowException(
            message=f"model {model_name} not found", error_code=ENDPOINT_NOT_FOUND
        )


def get_space_id_from_space_name(self, space_name: str) -> str:
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
            item for item in spaces["resources"] if item["entity"]["name"] == space_name
        )["metadata"]["id"]
    except StopIteration as _:
        raise MlflowException(
            message=f"space {space_name} not found", error_code=ENDPOINT_NOT_FOUND
        )


def get_model_id_from_model_details(client: APIClient, model_details: Dict) -> str:
    """_summary_

    Parameters
    ----------
    model_details : Dict
        _description_

    Returns
    -------
    str
        _description_
    """
    model_id = client.repository.get_model_id(model_details=model_details)
    return model_id


def get_deployment_id_from_deployment_details(
    client: APIClient, deployment_details: Dict
) -> str:
    """_summary_

    Parameters
    ----------
    client : APIClient
        _description_
    deployment_details : Dict
        _description_

    Returns
    -------
    str
        _description_
    """
    deployment_id = client.deployments.get_id(deployment_details=deployment_details)
    return deployment_id

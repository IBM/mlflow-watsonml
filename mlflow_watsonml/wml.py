from typing import Any, Dict

from ibm_watson_machine_learning.client import APIClient
from mlflow.exceptions import MlflowException

from mlflow_watsonml.utils import *


def store_model(
    client: APIClient,
    model_object: Any,
    software_spec_uid: str,
    name: str,
    model_description: str,
    model_type: str,
) -> Dict:
    """Store model_object in a WML repository

    Parameters
    ----------
    client : APIClient
        WML client
    model_object : Any
        artifact object
    software_spec_uid : str
        uid of software specification
    name : str
        name of the deployment
    model_description : str
        model description
    model_type : str
        type of model

    Returns
    -------
    Dict
        model details dictionary
    """
    model_props = {
        client.repository.ModelMetaNames.NAME: name,
        client.repository.ModelMetaNames.DESCRIPTION: model_description,
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid,
        client.repository.ModelMetaNames.TYPE: model_type,
    }

    try:
        model_details = client.repository.store_model(
            model=model_object,
            meta_props=model_props,
            training_data=None,
            training_target=None,
            feature_names=None,
            label_column_names=None,
        )

    except Exception as e:
        raise MlflowException(e)

    return model_details


def deploy_model(
    client: APIClient, name: str, model_id: str, batch: bool = False
) -> Dict:
    """Create a new WML deployment

    Parameters
    ----------
    client : APIClient
        WML client
    name : str
        name of the new deployment to create
    model_id : str
        UID of the model stored in WML repository
    batch : bool, optional
        whether to use batch or online method of deployment,
        by default False

    Returns
    -------
    Dict
        deployment details dictionary
    """

    if batch:
        deployment_props = {
            client.deployments.ConfigurationMetaNames.NAME: name,
            client.deployments.ConfigurationMetaNames.BATCH: {},
            client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {},
        }
    else:
        deployment_props = {
            client.deployments.ConfigurationMetaNames.NAME: name,
            client.deployments.ConfigurationMetaNames.ONLINE: {},
        }

    try:
        deployment_details = client.deployments.create(
            artifact_uid=model_id,
            meta_props=deployment_props,
            asynchronous=False,
        )

        deployment_details["name"] = deployment_details["entity"]["name"]

    except Exception as e:
        raise MlflowException(e)

    return deployment_details


def delete_deployment(client: APIClient, name: str):
    """Delete an existing deployment from WML

    Parameters
    ----------
    client : APIClient
        WML client
    name : str
        name of the deployment to delete
    """
    try:
        deployment_id = get_deployment_id_from_deployment_name(
            client=client, deployment_name=name
        )
        client.deployments.delete(deployment_uid=deployment_id)
    except Exception as e:
        raise MlflowException(e)


def delete_model(client: APIClient, name: str):
    """Delete a model from WML repository

    Parameters
    ----------
    client : APIClient
        WML client
    name : str
        name of the deployment to delete
    """
    try:
        model_id = get_model_id_from_model_name(client=client, model_name=name)
        client.repository.delete(artifact_uid=model_id)

    except Exception as e:
        raise MlflowException(e)


def update_model(client: APIClient):
    raise NotImplementedError()

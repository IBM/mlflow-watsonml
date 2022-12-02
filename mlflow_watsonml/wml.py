from typing import Any, Dict

from ibm_watson_machine_learning.client import APIClient
from mlflow.exceptions import MlflowException


def store_model(
    client: APIClient,
    model_object: Any,
    software_spec_uid: str,
    name: str,
    model_description: Dict,
    model_type: str,
) -> Dict:
    """_summary_

    Parameters
    ----------
    client : APIClient
        _description_
    model_object : Any
        _description_
    software_spec_uid : str
        _description_
    name : str
        _description_
    model_description : Dict
        _description_
    model_type : str
        _description_

    Returns
    -------
    Dict
        _description_

    Raises
    ------
    MlflowException
        _description_
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


def deploy_model(client: APIClient, name: str, model_id: str, batch: bool = False):
    """_summary_

    Parameters
    ----------
    client : APIClient
        _description_
    name : str
        _description_
    model_id : str
        _description_
    batch : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    MlflowException
        _description_
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

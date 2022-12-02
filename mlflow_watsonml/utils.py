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

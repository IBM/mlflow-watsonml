import logging
from typing import Any, Dict

from ibm_watson_machine_learning.client import APIClient
from mlflow.exceptions import MlflowException

from mlflow_watsonml.utils import *

LOGGER = logging.getLogger(__name__)


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
        LOGGER.info(model_details)
        LOGGER.info(f"Stored model {name} in the repository.")

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
        )

        deployment_details["name"] = deployment_details["entity"]["name"]

        LOGGER.info(deployment_details)
        LOGGER.info(f"Created {'batch' if batch else 'online'} deployment {name}")

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

        LOGGER.info(f"Deleted deployment {name} with id {deployment_id}.")
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

        LOGGER.info(f"Deleted model {name} with id {model_id} from the repository.")

    except Exception as e:
        raise MlflowException(e)


def update_model(client: APIClient):
    raise NotImplementedError()


def set_deployment_space(client: APIClient, deployment_space_name: str) -> APIClient:
    try:
        space_uid = get_space_id_from_space_name(
            client=client,
            space_name=deployment_space_name,
        )
        client.set.default_space(space_uid=space_uid)

        LOGGER.info(f"Set deployment space to {deployment_space_name}")

    except Exception as e:
        raise MlflowException(
            f"Failed to set deployment space {deployment_space_name}", f"{e}"
        )

    return client


def create_custom_software_spec(
    client: APIClient,
    name: str,
    base_sofware_spec: str,
    custom_packages: List[Dict[str, str]],
):
    if software_spec_exists(client=client, sw_spec=name):
        raise MlflowException(
            f"""Software spec {name} already exists. 
            Please delete the software spec or create one with another name."""
        )

    base_software_spec_id = client.software_specifications.get_id_by_name(
        base_sofware_spec
    )

    meta_prop_sw_spec = {
        client.software_specifications.ConfigurationMetaNames.NAME: name,
        client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {
            "guid": base_software_spec_id
        },
    }

    sw_spec_details = client.software_specifications.store(meta_props=meta_prop_sw_spec)
    software_spec_id = client.software_specifications.get_uid(sw_spec_details)

    for custom_package in custom_packages:
        meta_prop_pkg_extn = {
            client.package_extensions.ConfigurationMetaNames.NAME: custom_package[
                "name"
            ],
            client.package_extensions.ConfigurationMetaNames.TYPE: "pip_zip",
        }

        pkg_extn_details = client.package_extensions.store(
            meta_props=meta_prop_pkg_extn, file_path=custom_package["file"]
        )

        pkg_extn_id = client.package_extensions.get_id(pkg_extn_details)

        client.software_specifications.add_package_extension(
            software_spec_id, pkg_extn_id
        )

    return software_spec_id

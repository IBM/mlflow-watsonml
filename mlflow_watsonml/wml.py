import logging
from typing import Any, Dict, Optional, Tuple

from ibm_watson_machine_learning.client import APIClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import NOT_IMPLEMENTED

from mlflow_watsonml.store import *
from mlflow_watsonml.utils import *

LOGGER = logging.getLogger(__name__)


def deploy(
    client: APIClient,
    name: str,
    artifact_id: str,
    revision_id: str,
    batch: bool = False,
    environment_variables: Optional[Dict] = None,
    hardware_spec_id: Optional[str] = None,
) -> Dict:
    """Create a new WML deployment

    Parameters
    ----------
    client : APIClient
        WML client
    name : str
        name of the new deployment to create
    artifact_id : str
        UID of the model or function stored in WML repository
    batch : bool, optional
        whether to use batch or online method of deployment,
        by default False

    Returns
    -------
    Dict
        deployment details dictionary
    """
    if environment_variables is None:
        environment_variables = dict()

    if batch:
        deployment_props = {
            client.deployments.ConfigurationMetaNames.NAME: name,
            client.deployments.ConfigurationMetaNames.BATCH: {},
            client.deployments.ConfigurationMetaNames.CUSTOM: environment_variables,
            client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {
                "id": hardware_spec_id
            }
            if hardware_spec_id is not None
            else {},
            client.deployments.ConfigurationMetaNames.ASSET: {
                "id": artifact_id,
                "rev": revision_id,
            },
        }
    else:
        deployment_props = {
            client.deployments.ConfigurationMetaNames.NAME: name,
            client.deployments.ConfigurationMetaNames.CUSTOM: environment_variables,
            client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {
                "id": hardware_spec_id
            }
            if hardware_spec_id is not None
            else {},
            client.deployments.ConfigurationMetaNames.ONLINE: {},
            client.deployments.ConfigurationMetaNames.ASSET: {
                "id": artifact_id,
                "rev": revision_id,
            },
        }

    try:
        deployment_details = client.deployments.create(
            artifact_uid=artifact_id,
            meta_props=deployment_props,
        )

        deployment_details["name"] = deployment_details["metadata"]["name"]

        LOGGER.info(deployment_details)
        LOGGER.info(f"Created {'batch' if batch else 'online'} deployment - {name}")

    except Exception as e:
        raise MlflowException(e)

    return deployment_details


def delete_deployment(client: APIClient, name: str):
    """Delete an existing deployment from WML.
    This method deletes the deployment and all the artifacts associated with it.

    Parameters
    ----------
    client : APIClient
        WML client
    name : str
        name of the deployment to delete
    """
    try:
        deployment_details = get_deployment(client=client, name=name)

        deployment_id = client.deployments.get_id(deployment_details=deployment_details)
        client.deployments.delete(deployment_uid=deployment_id)
        LOGGER.info(f"Deleted deployment {name} with id {deployment_id}.")

        artifact_id = deployment_details["entity"]["asset"]["id"]
        artifact_details = client.repository.get_details(artifact_uid=artifact_id)
        client.repository.delete(artifact_uid=artifact_id)
        LOGGER.info(
            f"Deleted artifact {name} with id {artifact_id} from the repository."
        )

        software_spec_id = artifact_details["entity"]["software_spec"]["id"]
        software_spec_details = client.software_specifications.get_details(
            sw_spec_uid=software_spec_id
        )
        software_spec_name = software_spec_details["metadata"]["name"]

        if software_spec_name == f"{name}_sw_spec":
            client.software_specifications.delete(sw_spec_uid=software_spec_id)
            LOGGER.info(
                f"Deleted software specification {software_spec_name} with id {software_spec_id} from the repository."
            )

    except Exception as e:
        LOGGER.exception(e)
        raise MlflowException(e)


def update_deployment(
    client: APIClient,
    name: str,
    artifact_id: str,
    revision_id: str,
) -> Dict:
    deployment_id = get_deployment_id_from_deployment_name(
        client=client, deployment_name=name
    )
    metadata = {
        client.deployments.ConfigurationMetaNames.ASSET: {
            "id": artifact_id,
            "rev": revision_id,
        }
    }

    updated_deployment = client.deployments.update(
        deployment_uid=deployment_id, changes=metadata
    )

    LOGGER.info(updated_deployment)

    return updated_deployment


def store_or_update_artifact(
    client: APIClient,
    model_uri: str,
    artifact_name: str,
    flavor: str,
    software_spec_id: str,
    artifact_id: Optional[str] = None,
    environment_variables: Optional[Dict] = None,
) -> Tuple[str, str]:
    if flavor == "sklearn":
        artifact_id, revision_id = store_sklearn_artifact(
            client=client,
            model_uri=model_uri,
            artifact_name=artifact_name,
            software_spec_id=software_spec_id,
            artifact_id=artifact_id,
        )

    elif flavor == "onnx":
        artifact_id, revision_id = store_onnx_artifact(
            client=client,
            model_uri=model_uri,
            artifact_name=artifact_name,
            software_spec_id=software_spec_id,
            artifact_id=artifact_id,
        )

    elif flavor == "watson_nlp":
        artifact_id, revision_id = store_watson_nlp_artifact(
            client=client,
            model_uri=model_uri,
            artifact_name=artifact_name,
            software_spec_id=software_spec_id,
            artifact_id=artifact_id,
            config=environment_variables,
        )

    else:
        raise MlflowException(
            f"Flavor {flavor} is invalid or not implemented",
            error_code=NOT_IMPLEMENTED,
        )

    return (artifact_id, revision_id)


def create_custom_software_spec(
    client: APIClient,
    name: str,
    custom_packages: Optional[List[str]],
    conda_yaml: Optional[str] = None,
    rewrite: bool = False,
) -> str:
    if software_spec_exists(client=client, name=name):
        if rewrite:
            while software_spec_exists(client=client, name=name):
                software_spec_id = client.software_specifications.get_id_by_name(
                    sw_spec_name=name
                )
                client.software_specifications.delete(sw_spec_uid=software_spec_id)

                LOGGER.info(
                    f"Deleted software specification {name} with id {software_spec_id}"
                )
        else:
            LOGGER.warn(
                f"""Software spec {name} already exists."""
                """skipping software spec creation."""
                """restart with rewrite=True if software spec needs to be updated."""
            )

            software_spec_id = client.software_specifications.get_id_by_name(name)
            return software_spec_id

    try:
        base_software_spec_id = client.software_specifications.get_id_by_name(
            "runtime-22.2-py3.10"
        )

        meta_prop_sw_spec = {
            client.software_specifications.ConfigurationMetaNames.NAME: name,
            client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {
                "guid": base_software_spec_id
            },
        }

        sw_spec_details = client.software_specifications.store(
            meta_props=meta_prop_sw_spec
        )
        software_spec_id = client.software_specifications.get_id(sw_spec_details)

        if conda_yaml is not None:
            if not os.path.exists(conda_yaml):
                raise FileNotFoundError(f"conda.yaml file not found!")

            refine_conda_yaml(conda_yaml=conda_yaml)

            pkg_extn_name = f"{name}_conda_env"
            # pkg_extn_id =

            # if client.package_extensions.get_id_by_name(pkg_extn_name=pkg_extn_name) != "Not Found":

            meta_prop_pkg_extn = {
                client.package_extensions.ConfigurationMetaNames.NAME: pkg_extn_name,
                client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml",
            }

            pkg_extn_details = client.package_extensions.store(
                meta_props=meta_prop_pkg_extn,
                file_path=conda_yaml,
            )

            pkg_extn_id = client.package_extensions.get_uid(pkg_extn_details)

            client.software_specifications.add_package_extension(
                software_spec_id, pkg_extn_id
            )

        if custom_packages is not None:
            for custom_package in custom_packages:
                if not is_zipfile(custom_package):
                    raise MlflowException(f"{custom_package} is not a valid zip file.")

                pkg_name = os.path.splitext(os.path.basename(custom_package))[0]

                meta_prop_pkg_extn = {
                    client.package_extensions.ConfigurationMetaNames.NAME: pkg_name,
                    client.package_extensions.ConfigurationMetaNames.TYPE: "pip_zip",
                }

                pkg_extn_details = client.package_extensions.store(
                    meta_props=meta_prop_pkg_extn, file_path=custom_package
                )

                pkg_extn_id = client.package_extensions.get_id(pkg_extn_details)

                client.software_specifications.add_package_extension(
                    software_spec_id, pkg_extn_id
                )

    except Exception as e:
        LOGGER.exception(e)

        raise MlflowException(e)

    LOGGER.info(
        f"Successfully created {name} software specification with ID {software_spec_id}"
    )

    return software_spec_id

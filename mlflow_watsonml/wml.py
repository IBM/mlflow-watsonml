import logging
from typing import Any, Dict, Optional, Tuple

from ibm_watson_machine_learning.client import APIClient
from mlflow.exceptions import MlflowException

from mlflow_watsonml.utils import *

LOGGER = logging.getLogger(__name__)


def deploy(
    client: APIClient,
    name: str,
    artifact_id: str,
    revision_id: str,
    batch: bool = False,
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

    if batch:
        deployment_props = {
            client.deployments.ConfigurationMetaNames.NAME: name,
            client.deployments.ConfigurationMetaNames.BATCH: {},
            client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {},
            client.deployments.ConfigurationMetaNames.ASSET: {
                "id": artifact_id,
                "rev": revision_id,
            },
        }
    else:
        deployment_props = {
            client.deployments.ConfigurationMetaNames.NAME: name,
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
        client.software_specifications.delete(sw_spec_uid=software_spec_id)
        LOGGER.info(
            f"Deleted software specification {software_spec_details['metadata']['name']} with id {software_spec_id} from the repository."
        )

    except Exception as e:
        LOGGER.exception(e)
        raise MlflowException(e)


def update_model(
    client: APIClient,
    deployment_name: str,
    updated_model_config: Dict = {},
    updated_model_object: Optional[Any] = None,
) -> Tuple[str, str]:
    try:
        deployment_details = get_deployment(client=client, name=deployment_name)

        model_id = deployment_details["entity"]["asset"]["id"]
        model_rev = int(deployment_details["entity"]["asset"]["rev"]) + 1

        if updated_model_object is not None:
            updated_model_config[
                client.repository.ModelMetaNames.NAME
            ] = f"{deployment_name}_v{model_rev}"

        updated_model_details = client.repository.update_model(
            model_uid=model_id,
            updated_meta_props=updated_model_config,
            update_model=updated_model_object,
        )

        updated_model_id = client.repository.get_model_id(updated_model_details)

        revised_model_details = client.repository.create_model_revision(
            updated_model_id
        )
        revision_id = revised_model_details["metadata"]["rev"]

    except Exception as e:
        raise MlflowException(e)

    return (updated_model_id, revision_id)


def update_deployment(
    client: APIClient,
    name: str,
    model_id: str,
    revision_id: str,
):
    deployment_id = get_deployment_id_from_deployment_name(
        client=client, deployment_name=name
    )
    metadata = {
        client.deployments.ConfigurationMetaNames.ASSET: {
            "id": model_id,
            "rev": revision_id,
        }
    }

    updated_deployment = client.deployments.update(
        deployment_uid=deployment_id, changes=metadata
    )

    LOGGER.info(updated_deployment)


def create_custom_software_spec(
    client: APIClient,
    name: str,
    custom_packages: Optional[List[Dict[str, str]]],
    conda_yaml: Optional[str] = None,
    rewrite: bool = False,
) -> str:
    if software_spec_exists(client=client, name=name):
        if rewrite:
            while software_spec_exists(client=client, name=name):
                delete_software_spec(client=client, name=name)
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
                if not is_zipfile(custom_package["file"]):
                    raise MlflowException(
                        f"{custom_package['file']} is not a valid zip file."
                    )

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

    except Exception as e:
        LOGGER.exception(e)

        raise MlflowException(e)

    LOGGER.info(
        f"Successfully created {name} software specification with ID {software_spec_id}"
    )

    return software_spec_id

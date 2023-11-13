import logging
import os
from types import FunctionType
from typing import Any, Dict, Optional, Tuple

import mlflow
from ibm_watson_machine_learning.client import APIClient
from mlflow.exceptions import MlflowException

LOGGER = logging.getLogger(__name__)


def store_or_update_model(
    client: APIClient,
    model_object: Any,
    model_name: str,
    model_type: str,
    software_spec_uid: str,
    model_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Store or update a model object in a WML repository

    Parameters
    ----------
    client : APIClient
        WML client
    model_object : Any
        model object
    model_name : str
        name of the model
    model_type : str
        type of model
    software_spec_uid : str
        UID of software specification
    model_id : str, optional
        asset id of the model to be updated
        by default None

    Returns
    -------
    Tuple[str, str]
        model id, model revision id
    """

    try:
        if model_id is None:
            model_props = {
                client.repository.ModelMetaNames.NAME: model_name,
                client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid,
                client.repository.ModelMetaNames.TYPE: model_type,
            }
            model_details = client.repository.store_model(
                model=model_object,
                meta_props=model_props,
                training_data=None,
                training_target=None,
                feature_names=None,
                label_column_names=None,
            )
            LOGGER.info(model_details)
            LOGGER.info(f"Stored model {model_name} in the repository.")

            model_id = client.repository.get_model_id(model_details=model_details)
        else:
            model_props = {
                client.repository.ModelMetaNames.NAME: model_name,
                client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid,
                client.repository.ModelMetaNames.TYPE: model_type,
            }
            model_details = client.repository.update_model(
                model_uid=model_id,
                updated_meta_props=model_props,
                update_model=model_object,
            )
            LOGGER.info(model_details)
            LOGGER.info(f"Updated model {model_name} in the repository.")

            model_id = client.repository.get_model_id(model_details=model_details)

        revision_details = client.repository.create_model_revision(model_uid=model_id)

        rev_id = revision_details["metadata"]["rev"]
        LOGGER.info(revision_details)
        LOGGER.info(
            f"Created model revision for model {model_name} and version {rev_id}"
        )

    except Exception as e:
        raise MlflowException(e)

    return (model_id, rev_id)


def store_or_update_function(
    client: APIClient,
    deployable_function: FunctionType,
    function_name: str,
    software_spec_uid: str,
    function_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Store or update a python function in WML repository

    Parameters
    ----------
    client : APIClient
        WML client
    deployable_function : FunctionType
        function to deploy
    function_name : str
        name of the function
    software_spec_uid : str
        software specification id
    function_id: str, optional
        asset id of the function to be updated
        by default None

    Returns
    -------
    Tuple[str, str]
        function id and function revisin id
    """
    try:
        if function_id is None:
            metaprops = {
                client.repository.FunctionMetaNames.NAME: function_name,
                client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID: software_spec_uid,
            }
            function_details = client.repository.store_function(
                function=deployable_function,
                meta_props=metaprops,
            )
            LOGGER.info(function_details)
            LOGGER.info(f"Stored function {function_name} in the repository.")

            function_id = client.repository.get_function_id(
                function_details=function_details
            )
        else:
            metaprops = {
                client.repository.FunctionMetaNames.NAME: function_name,
            }
            function_details = client.repository.update_function(
                function_uid=function_id,
                changes=metaprops,
                update_function=deployable_function,
            )
            LOGGER.info(function_details)
            LOGGER.info(f"Updated function {function_name} in the repository.")

            function_id = client.repository.get_function_id(
                function_details=function_details
            )

        revision_details = client.repository.create_function_revision(
            function_uid=function_id
        )
        rev_id = revision_details["metadata"]["rev"]
        LOGGER.info(revision_details)
        LOGGER.info(
            f"Created function revision for function {function_name} and version {rev_id}"
        )
    except Exception as e:
        LOGGER.exception(e)
        raise MlflowException(e)

    return (function_id, rev_id)


def store_onnx_artifact(
    client: APIClient,
    model_uri: str,
    artifact_name: str,
    software_spec_id: str,
    artifact_id: Optional[str] = None,
) -> Tuple[str, str]:
    """store onnx artifact in WML

    Parameters
    ----------
    client : APIClient
        WML client
    model_uri : str
        model URI
    artifact_name : str
        name of the artifact
    software_spec_id : str
        id of software specification
    artifact_id : Optional[str], optional
        artifact id of the stored model, by default None

    Returns
    -------
    Tuple[str, str]
        model id, revision id
    """

    # the args have to be passed as default value in the scorer
    def deployable_onnx_scorer(artifact_uri=model_uri):
        import os
        import tempfile

        import mlflow
        import onnx  # type: ignore
        from onnxruntime import InferenceSession  # type: ignore

        def score(payload: dict):
            artifact_dir = os.path.join(tempfile.gettempdir(), "artifacts")

            # `download_artifacts` returns the local path if it's already been downloaded
            artifact_file = mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri, dst_path=artifact_dir
            )
            model_file = os.path.join(artifact_file, "model.onnx")
            onnx.checker.check_model(model_file)  # type: ignore
            model = onnx.load(model_file)
            input_name = model.graph.input[0].name

            scoring_output = {"predictions": []}

            sess = InferenceSession(model.SerializeToString())

            for data in payload["input_data"]:
                values = data.get("values")
                # fields = data.get("fields")
                predictions = sess.run(None, {input_name: values})[0]

                scoring_output["predictions"].append({"values": predictions.tolist()})

            return scoring_output

        return score

    function_id, rev_id = store_or_update_function(
        client=client,
        deployable_function=deployable_onnx_scorer,
        function_name=artifact_name,
        software_spec_uid=software_spec_id,
        function_id=artifact_id,
    )

    return (function_id, rev_id)


def store_sklearn_artifact(
    client: APIClient,
    model_uri: str,
    artifact_name: str,
    software_spec_id: str,
    artifact_id: Optional[str] = None,
) -> Tuple[str, str]:
    """store sklearn artifact in WML

    Parameters
    ----------
    client : APIClient
        WML client
    model_uri : str
        model URI
    artifact_name : str
        name of the artifact
    software_spec_id : str
        id of software specification
    artifact_id : Optional[str], optional
        artifact id of the stored model, by default None

    Returns
    -------
    Tuple[str, str]
        model id, revision id
    """
    model_object = mlflow.sklearn.load_model(model_uri=model_uri)
    model_id, rev_id = store_or_update_model(
        client=client,
        model_object=model_object,
        model_name=artifact_name,
        model_type="scikit-learn_1.1",
        software_spec_uid=software_spec_id,
        model_id=artifact_id,
    )

    return (model_id, rev_id)


def store_watson_nlp_artifact(
    client: APIClient,
    model_uri: str,
    artifact_name: str,
    software_spec_id: str,
    artifact_id: Optional[str] = None,
    config: Optional[Dict] = None,
) -> Tuple[str, str]:
    """store watson nlp artifact in WML

    Parameters
    ----------
    client : APIClient
        WML client
    model_uri : str
        model URI
    artifact_name : str
        name of the artifact
    software_spec_id : str
        id of software specification
    artifact_id : Optional[str], optional
        artifact id of the stored model, by default None

    Returns
    -------
    Tuple[str, str]
        model id, revision id
    """

    # the args have to be passed as default value in the scorer
    def deployable_watson_nlp_scorer(artifact_uri=model_uri, config=config):
        import os
        import tempfile

        import mlflow
        import watson_nlp  # type: ignore

        for key, val in config.items():  # type: ignore
            os.environ[key] = val

        def score(payload: dict):
            artifact_dir = os.path.join(tempfile.gettempdir(), "artifacts")

            # `download_artifacts` returns the local path if it's already been downloaded
            artifact_file = mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri, dst_path=artifact_dir
            )

            model = watson_nlp.load(artifact_file)

            scoring_output = {"predictions": []}

            for data in payload["input_data"]:
                values = data.get("values")
                # fields = data.get("fields")
                predictions = model.run_batch(values)
                predictions = [prediction.to_dict() for prediction in predictions]

                scoring_output["predictions"].append({"values": predictions})

            return scoring_output

        return score

    function_id, rev_id = store_or_update_function(
        client=client,
        deployable_function=deployable_watson_nlp_scorer,
        function_name=artifact_name,
        software_spec_uid=software_spec_id,
        function_id=artifact_id,
    )

    return (function_id, rev_id)

import logging
import os
from types import FunctionType
from typing import Any, Dict, Tuple

from ibm_watson_machine_learning.client import APIClient
from mlflow.exceptions import MlflowException

LOGGER = logging.getLogger(__name__)


# TODO: implement logic to make sure the environment variables are set
def get_s3_creds():
    return {
        "MLFLOW_S3_ENDPOINT_URL": os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
    }


def store_model(
    client: APIClient,
    model_object: Any,
    software_spec_uid: str,
    model_name: str,
    model_type: str,
) -> Tuple[str, str]:
    """Store model_object in a WML repository

    Parameters
    ----------
    client : APIClient
        WML client
    model_object : Any
        artifact object
    software_spec_uid : str
        uid of software specification
    model_name : str
        name of the model
    model_type : str
        type of model

    Returns
    -------
    Tuple[str, str]
        model id, model revision id
    """
    model_props = {
        client.repository.ModelMetaNames.NAME: model_name,
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
        LOGGER.info(f"Stored model {model_name} in the repository.")

        model_id = client.repository.get_model_id(model_details=model_details)

        revision_details = client.repository.create_model_revision(model_uid=model_id)

        rev_id = revision_details["metadata"].get("rev")
        LOGGER.info(revision_details)
        LOGGER.info(
            f"Created model revision for model {model_name} and version {rev_id}"
        )

    except Exception as e:
        raise MlflowException(e)

    return (model_id, rev_id)


def store_function(
    client: APIClient,
    deployable_function: FunctionType,
    function_name: str,
    software_spec_uid: str,
) -> Tuple[str, str]:
    """Store function in WML repository

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

    Returns
    -------
    Tuple[str, str]
        function id and function revisin id
    """
    metaprops = {
        client.repository.FunctionMetaNames.NAME: function_name,
        client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID: software_spec_uid,
    }

    try:
        function_details = client.repository.store_function(
            function=deployable_function,
            meta_props=metaprops,
        )
        LOGGER.info(function_details)
        LOGGER.info(f"Stored function {function_name} in the repository.")

        function_id = client.repository.get_function_id(
            function_details=function_details
        )

        revision_details = client.repository.create_function_revision(
            function_uid=client.repository.get_function_id(
                function_details=function_details
            )
        )
        rev_id = revision_details["metadata"].get("rev")
        LOGGER.info(revision_details)
        LOGGER.info(
            f"Created function revision for function {function_name} and version {rev_id}"
        )
    except Exception as e:
        raise MlflowException(e)

    return (function_id, rev_id)


def store_onnx_artifact(
    client: APIClient, model_uri: str, artifact_name: str, software_spec_id: str
):
    config = get_s3_creds()

    # the args have to be passed as default value in the scorer
    def deployable_onnx_scorer(artifact_uri=model_uri, config=config):
        import os
        import tempfile

        import mlflow
        import onnx
        from onnxruntime import InferenceSession

        def score(payload: dict):
            for key, value in config.items():
                os.environ[key] = value

            artifact_dir = os.path.join(tempfile.gettempdir(), "artifacts")
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

    function_id, rev_id = store_function(
        client=client,
        deployable_function=deployable_onnx_scorer,
        function_name=artifact_name,
        software_spec_uid=software_spec_id,
    )

    return (function_id, rev_id)


def store_sklearn_artifact(
    client: APIClient, model_uri: str, artifact_name: str, software_spec_id: str
):
    config = get_s3_creds()

    # the args have to be passed as default value in the scorer
    def deployable_sklearn_scorer(model_uri=model_uri, config=config):
        import os

        import mlflow

        def score(payload: dict):
            for key, value in config.items():
                os.environ[key] = value

            model = mlflow.sklearn.load_model(model_uri=model_uri)

            if model is None:
                return {"predictions": [{"values": "no model found"}]}

            scoring_output = {"predictions": []}

            for data in payload["input_data"]:
                values = data.get("values")
                # fields = data.get("fields")
                predictions = model.predict(values)

                scoring_output["predictions"].append({"values": predictions.tolist()})

            return scoring_output

        return score

    function_id, rev_id = store_function(
        client=client,
        deployable_function=deployable_sklearn_scorer,
        function_name=artifact_name,
        software_spec_uid=software_spec_id,
    )

    return (function_id, rev_id)

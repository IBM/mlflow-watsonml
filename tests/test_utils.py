import zipfile

import pytest
from mlflow import MlflowException
from pytest import LogCaptureFixture, MonkeyPatch
from resources.mock.mock_client import MockAPIClient

import mlflow_watsonml.deploy
from mlflow_watsonml.deploy import WatsonMLDeploymentClient
from mlflow_watsonml.utils import *

MOCK_WML_CREDENTIALS = {
    "username": "user",
    "apikey": "correct_api_key",
    "url": "https://url",
    "instance_id": "wml",
    "version": "1.0",
}


@pytest.fixture(autouse=True)
def mock_client(monkeypatch: MonkeyPatch):
    # Mock the APIClient
    monkeypatch.setattr(mlflow_watsonml.deploy, "APIClient", MockAPIClient)


def test_list_artifacts():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    artifacts = list_artifacts(client=client)

    assert isinstance(artifacts, list)
    assert len(artifacts) == 3

    for artifact in artifacts:
        assert isinstance(artifact, dict)
        assert "name" in artifact.keys()


def test_get_artifact_id_from_artifact_name_success():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    artifact_id = get_artifact_id_from_artifact_name(
        client=client, artifact_name="artifact_1"
    )

    assert artifact_id == "id_of_artifact_1"


def test_get_artifact_id_from_artifact_name_exception(caplog: LogCaptureFixture):
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    with pytest.raises(MlflowException):
        artifact_id = get_artifact_id_from_artifact_name(
            client=client, artifact_name="artifact_01"
        )

    assert "artifact artifact_01 not found" in caplog.text


def test_artifact_exists():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    assert artifact_exists(client=client, name="artifact_1")
    assert not artifact_exists(client=client, name="artifact_01")


def test_list_deployments():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    deployments = list_deployments(client=client)

    assert isinstance(deployments, list)
    assert len(deployments) == 2

    for deployment in deployments:
        assert isinstance(deployment, dict)
        assert "name" in deployment.keys()


def test_get_deployment_success():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    deployment = get_deployment(client=client, name="deployment_1")

    assert isinstance(deployment, dict)
    assert "name" in deployment.keys()
    assert "metadata" in deployment.keys()
    assert isinstance(deployment["metadata"], dict)
    assert "id" in deployment["metadata"].keys()


def test_get_deployment_exception(caplog: LogCaptureFixture):
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    with pytest.raises(MlflowException):
        deployment = get_deployment(client=client, name="deployment_01")

    assert "no deployment by the name deployment_01 exists" in caplog.text


def test_get_deployment_id_from_deployment_name_success():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    deployment_id = get_deployment_id_from_deployment_name(
        client=client, deployment_name="deployment_1"
    )

    assert deployment_id == "id_of_deployment_1"


def test_get_deployment_id_from_deployment_name_exception(caplog: LogCaptureFixture):
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    with pytest.raises(MlflowException):
        deployment_id = get_deployment_id_from_deployment_name(
            client=client, deployment_name="deployment_01"
        )

    assert "no deployment by the name deployment_01 exists" in caplog.text


def test_deployment_exists():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    assert deployment_exists(client=client, name="deployment_1")
    assert not deployment_exists(client=client, name="deployment_01")


def test_get_space_id_from_space_name_success():
    client = mlflow_watsonml.deploy.APIClient(wml_credentials=MOCK_WML_CREDENTIALS)

    space_id = get_space_id_from_space_name(client=client, space_name="space_1")

    assert space_id == "id_of_space_1"


@pytest.fixture
def zip_file_path(tmp_path):
    # Create a temporary zip file for testing
    file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(file_path, "w") as zf:
        zf.writestr("file1.txt", "Hello, World!")
        zf.writestr("file2.txt", "artifact Factory")

    yield str(file_path)

    # Clean up the temporary file
    os.remove(file_path)


def test_is_zipfile_valid(zip_file_path):
    assert is_zipfile(zip_file_path) is True


def test_is_zipfile_invalid(zip_file_path):
    # Modify the file extension to make it invalid
    invalid_file_path = zip_file_path.replace(".zip", ".txt")
    assert is_zipfile(invalid_file_path) is False


def test_is_zipfile_nonexistent():
    non_existent_file_path = "/path/to/nonexistent.zip"
    assert is_zipfile(non_existent_file_path) is False


def test_is_zipfile_corrupt_zip(zip_file_path):
    # Overwrite the zip file with a corrupt file
    with open(zip_file_path, "w") as f:
        f.write("corrupt file data")

    assert is_zipfile(zip_file_path) is False


def test_get_software_spec_from_deployment_name_success():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    sw_spec = get_software_spec_from_deployment_name(
        client=client, deployment_name="deployment_1"
    )

    assert sw_spec == "id_of_sw_spec_1"


def test_get_software_spec_from_deployment_name_exception(caplog: LogCaptureFixture):
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    with pytest.raises(MlflowException):
        sw_spec = get_software_spec_from_deployment_name(
            client=client, deployment_name="deployment_3"
        )

    assert f"no deployment by the name deployment_3 exists"

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
    "apikey": "abcdefghijkl",
    "url": "https://url",
    "instance_id": "wml",
    "version": "1.0",
}


@pytest.fixture(autouse=True)
def mock_client(monkeypatch: MonkeyPatch):
    # Mock the APIClient
    monkeypatch.setattr(mlflow_watsonml.deploy, "APIClient", MockAPIClient)


def test_list_models():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    models = list_models(client=client)

    assert isinstance(models, list)
    assert len(models) == 3

    for model in models:
        assert isinstance(model, dict)
        assert "name" in model.keys()


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


def test_get_model_id_from_model_name_success():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    model_id = get_model_id_from_model_name(client=client, model_name="model_1")

    assert model_id == "id_of_model_1"


def test_get_model_id_from_model_name_exception(caplog: LogCaptureFixture):
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    with pytest.raises(MlflowException):
        model_id = get_model_id_from_model_name(client=client, model_name="model_01")

    assert "model model_01 not found" in caplog.text


def test_get_space_id_from_space_name_success():
    client = mlflow_watsonml.deploy.APIClient(wml_credentials=MOCK_WML_CREDENTIALS)

    space_id = get_space_id_from_space_name(client=client, space_name="space_1")

    assert space_id == "id_of_space_1"


def test_get_space_id_from_space_name_exception(caplog: LogCaptureFixture):
    client = mlflow_watsonml.deploy.APIClient(wml_credentials=MOCK_WML_CREDENTIALS)

    with pytest.raises(MlflowException):
        space_id = get_space_id_from_space_name(client=client, space_name="space_01")

    assert "space space_01 not found" in caplog.text


def test_get_deployment_id_from_deployment_details_success():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    deployment_id = get_deployment_id_from_deployment_details(
        client=client,
        deployment_details={
            "metadata": {"name": "deployment_1", "id": "id_of_deployment_1"}
        },
    )

    assert deployment_id == "id_of_deployment_1"


def test_get_model_id_from_model_details_success():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    model_id = get_model_id_from_model_details(
        client=client,
        model_details={"metadata": {"name": "model_1", "id": "id_of_model_1"}},
    )

    assert model_id == "id_of_model_1"


def test_deployment_exists():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    assert deployment_exists(client=client, name="deployment_1")
    assert not deployment_exists(client=client, name="deployment_01")


def test_model_exists():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    assert model_exists(client=client, name="model_1")
    assert not model_exists(client=client, name="model_01")


def test_list_endpoints():
    client = mlflow_watsonml.deploy.APIClient(wml_credentials=MOCK_WML_CREDENTIALS)

    endpoints = list_endpoints(client=client)

    assert isinstance(endpoints, list)
    assert len(endpoints) == 2
    assert endpoints[0]["entity"]["name"] == "space_1"
    assert endpoints[0]["metadata"]["id"] == "id_of_space_1"
    assert endpoints[1]["entity"]["name"] == "space_2"
    assert endpoints[1]["metadata"]["id"] == "id_of_space_2"


def test_get_endpoint_success():
    client = mlflow_watsonml.deploy.APIClient(wml_credentials=MOCK_WML_CREDENTIALS)

    space = get_endpoint(client=client, endpoint="space_1")
    assert isinstance(space, dict)
    assert space["entity"]["name"] == "space_1"
    assert space["metadata"]["id"] == "id_of_space_1"


def test_get_endpoint_exception(caplog: LogCaptureFixture):
    client = mlflow_watsonml.deploy.APIClient(wml_credentials=MOCK_WML_CREDENTIALS)

    with pytest.raises(Exception):
        space = get_endpoint(client=client, endpoint="space_01")

    assert "space space_01 not found" in caplog.text


def test_list_software_specs():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    sw_specs = list_software_specs(client=client)

    assert isinstance(sw_specs, list)
    assert len(sw_specs) == 2
    assert sw_specs[0]["metadata"]["name"] == "sw_spec_1"
    assert sw_specs[0]["metadata"]["asset_id"] == "id_of_sw_spec_1"
    assert sw_specs[1]["metadata"]["name"] == "sw_spec_2"
    assert sw_specs[1]["metadata"]["asset_id"] == "id_of_sw_spec_2"


def test_get_sw_spec_success():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    sw_spec = get_software_spec(client=client, name="sw_spec_1")

    assert isinstance(sw_spec, str)
    assert sw_spec == "id_of_sw_spec_1"


def test_get_sw_spec_exception(caplog: LogCaptureFixture):
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    with pytest.raises(MlflowException):
        sw_spec = get_software_spec(client=client, name="sw_spec_01")

    assert "Software Specifiction - sw_spec_01 not found" in caplog.text


def test_software_spec_exists():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )

    assert software_spec_exists(client=client, name="sw_spec_1")
    assert not software_spec_exists(client=client, name="sw_spec_01")


def test_delete_sw_spec(caplog: LogCaptureFixture):
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS).get_wml_client(
        endpoint="space_1"
    )
    assert software_spec_exists(client=client, name="sw_spec_1")

    delete_sw_spec(client=client, name="sw_spec_1")

    assert not software_spec_exists(client=client, name="sw_spec_1")


@pytest.fixture
def zip_file_path(tmp_path):
    # Create a temporary zip file for testing
    file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(file_path, "w") as zf:
        zf.writestr("file1.txt", "Hello, World!")
        zf.writestr("file2.txt", "Model Factory")

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


def test_load_model_success(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda model_uri: 0)

    model_obj, model_type = load_model(model_uri="some_uri", flavor="sklearn")

    assert model_obj == 0
    assert model_type == "scikit-learn_1.1"


def test_load_model_exception(monkeypatch: MonkeyPatch, caplog: LogCaptureFixture):
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda model_uri: 0)

    with pytest.raises(NotImplementedError):
        _, _ = load_model(model_uri="some_uri", flavor="xyz")

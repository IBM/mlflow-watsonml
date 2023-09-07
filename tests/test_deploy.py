import pytest
from mlflow import MlflowException
from pytest import LogCaptureFixture, MonkeyPatch
from resources.mock.mock_client import MockAPIClient

import mlflow_watsonml.deploy
from mlflow_watsonml.deploy import WatsonMLDeploymentClient

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


def test_connect_success():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS)

    # Assert that the _wml_client attribute is set
    assert isinstance(client._wml_client, MockAPIClient)


def test_connect_exception(caplog: LogCaptureFixture):
    # Call the connect method with mock credentials
    mock_wml_credentials = MOCK_WML_CREDENTIALS.copy()
    mock_wml_credentials["apikey"] = "incorrect_api_key"

    assert MOCK_WML_CREDENTIALS["apikey"] != "incorrect_api_key"

    with pytest.raises(Exception):
        _ = WatsonMLDeploymentClient(config=mock_wml_credentials)

    # Assert that the exception was logged
    assert "Connection Failed!" in caplog.text


def test_get_wml_client_success():
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS)

    wml_client = client.get_wml_client(endpoint="space_1")

    assert isinstance(wml_client, MockAPIClient)


def test_get_wml_client_exception(caplog: LogCaptureFixture):
    client = WatsonMLDeploymentClient(config=MOCK_WML_CREDENTIALS)

    with pytest.raises(MlflowException):
        _ = client.get_wml_client(endpoint="space_3")

    assert "space space_3 not found" in caplog.text


def test_create_deployment_success(monkeypatch: MonkeyPatch):
    ...


def test_create_deployment_exception(caplog: LogCaptureFixture):
    ...


def test_update_deployment_success():
    ...


def test_update_deployment_exception(caplog: LogCaptureFixture):
    ...


def test_delete_deployment_success():
    ...


def test_delete_deployment_exception(caplog: LogCaptureFixture):
    ...


def test_list_deployments_success():
    ...


def test_list_deployments_exception(caplog: LogCaptureFixture):
    ...


def test_get_deployment_success():
    ...


def test_get_deployment_exception(caplog: LogCaptureFixture):
    ...


def test_predict_success():
    ...


def test_predict_exception(caplog: LogCaptureFixture):
    ...

import json
import os

import pytest
from mlflow import MlflowException

from mlflow_watsonml.config import Config


@pytest.fixture(scope="session")
def create_correct_credentials(tmp_path_factory: pytest.TempPathFactory):
    credentials = {
        "username": "username",
        "apikey": "apikey",
        "url": "url",
        "instance_id": "openshift",
        "version": "4.0",
    }

    credentials_path = tmp_path_factory.mktemp("credentials") / "wml_credentials.json"
    with open(credentials_path, "w") as f:
        json.dump(credentials, f)

    return credentials_path


def test_config_input_arg_pass():
    input = dict()
    input["apikey"] = "apikey"
    input["location"] = "location"
    input["url"] = "url"

    config = Config(config=input)

    assert config["wml_credentials"] == {"apikey": "apikey", "url": "url"}


def test_config_input_arg_fail():
    input = dict()
    input["apikey"] = "apikey"

    with pytest.raises(MlflowException):
        _ = Config(config=input)


def test_config_input_env_var_pass(create_correct_credentials):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WML_CREDENTIALS_FILE", str(create_correct_credentials))

        config = Config()

        assert config["wml_credentials"] == {
            "username": "username",
            "apikey": "apikey",
            "url": "url",
            "instance_id": "openshift",
            "version": "4.0",
        }


def test_config_input_env_var_fail():
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("WML_CREDENTIALS_FILE", raising=False)
        mp.delenv("URL", raising=False)
        mp.setenv("APIKEY", "apikey")

        with pytest.raises(MlflowException):
            _ = Config()

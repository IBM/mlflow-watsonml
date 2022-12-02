import logging
import os
import sys
import zipfile
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from ibm_watson_machine_learning.client import APIClient
from mlflow.deployments import BaseDeploymentClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (ENDPOINT_NOT_FOUND,
                                          INVALID_PARAMETER_VALUE)


def target_help():
    # TODO: Improve
    help_string = (
        "The mlflow-watsonml plugin integrates IBM WatsonML "
        "with the MLFlow deployments API.\n\n"
        "Before using this plugin, you must set up a json file "
        "containing WML credentials and create an environment variable "
        "along with deployment space name. "
    )
    return help_string


def run_local(name, model_uri, flavor=None, config=None):
    # TODO: implement
    raise MlflowException("mlflow-watsonml does not currently support run_local.")

    
class WatsonMLDeploymentClient(BaseDeploymentClient):
    def __init__(self, target_uri):
        super().__init__(target_uri)
    
    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        return super().create_deployment(name, model_uri, flavor, config, endpoint)
    
    def delete_deployment(self, name, config=None, endpoint=None):
        return super().delete_deployment(name, config, endpoint)
    
    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        return super().update_deployment(name, model_uri, flavor, config, endpoint)
    
    def list_deployments(self, endpoint=None):
        return super().list_deployments(endpoint)
    
    def get_deployment(self, name, endpoint=None):
        return super().get_deployment(name, endpoint)
    
    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        return super().predict(deployment_name, inputs, endpoint)
    
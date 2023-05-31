import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow import MlflowException
from mlflow.deployments import get_deploy_client
from mlflow.models.signature import infer_signature
from sklearn.linear_model import ElasticNet, LogisticRegression

from mlflow_watsonml.deploy import WatsonMLDeploymentClient


def get_linear_lr():
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    predictions = lr.predict(X)
    signature = infer_signature(X, predictions)
    mlflow.sklearn.log_model(lr, "model", signature=signature)
    run_id = mlflow.active_run().info.run_uuid
    print("Model saved in run %s" % run_id)
    return run_id


def get_elastic_net():
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = ElasticNet(alpha=0.5)
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    predictions = lr.predict(X)
    signature = infer_signature(X, predictions)
    mlflow.sklearn.log_model(lr, "model", signature=signature)
    run_id = mlflow.active_run().info.run_uuid
    print("Model saved in run %s" % run_id)
    return run_id


def test_wml_deployment():
    """
    1. test whether client is able to connect to WML
    2. check if the deployment exists, delete if it does
    3. create a new deployment with linear regression
    4. score with the created deployment
    5. update the deployment with elastic net
    6. score with the updated deployment
    7. delete the deployment
    """
    # connect to wml client
    client: WatsonMLDeploymentClient = get_deploy_client("watsonml")
    assert client is not None

    deployment_name = "test_deployment"

    # train a linear regression model and log it
    run_id = get_linear_lr()
    model_uri = f"runs:/{run_id}/model"

    # delete the deployment if it exists
    client.delete_deployment(deployment_name)

    with pytest.raises(MlflowException):
        client.get_deployment(deployment_name)

    # create a new deployment
    client.create_deployment(
        name=deployment_name, model_uri=model_uri, flavor="sklearn"
    )

    client.get_deployment(deployment_name)

    # train a linear regression model and log it
    run_id = get_elastic_net()
    model_uri = f"runs:/{run_id}/model"

    client.update_deployment(deployment_name, model_uri=model_uri, flavor="sklearn")

    client.get_deployment(deployment_name)

    # delete deplyment
    client.delete_deployment(deployment_name)

    with pytest.raises(MlflowException):
        client.get_deployment(deployment_name)

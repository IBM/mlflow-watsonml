import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.deployments import get_deploy_client
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression

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


def test_create_deployment():
    client: WatsonMLDeploymentClient = get_deploy_client("watsonml")

    run_id = get_linear_lr()
    model_uri = f"runs:/{run_id}/model"

    client.delete_deployment("test_deployment")

    client.create_deployment(
        name="test_deployment", model_uri=model_uri, flavor="sklearn"
    )

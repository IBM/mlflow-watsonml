from mlflow.deployments import get_deploy_client


def test_init():
    client = get_deploy_client("watsonml")

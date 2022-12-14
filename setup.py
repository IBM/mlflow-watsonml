from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mlflow-watsonml",
    version="0.0.1",
    description="WatsonML MLflow deployment plugin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/mlflow-watsonml",
    packages=find_packages(),
    entry_points={"mlflow.deployments": "watsonml=mlflow_watsonml.deploy"},
)

from setuptools import find_packages, setup

setup(
    name="mlflow-watsonml",
    version="0.0.1",
    packages=find_packages(),
    entry_points={"mlflow.deployments": "watsonml=mlflow_watsonml.deploy"},
)

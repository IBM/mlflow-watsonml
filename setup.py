from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

with open("requirements.txt", "r") as fh:
    install_requires = fh.readlines()

setup(
    name="mlflow-watsonml",
    version=version,
    description="WatsonML MLflow deployment plugin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/mlflow-watsonml",
    packages=["mlflow_watsonml"],
    author="IBM AI Model Factory team",
    author_email="dhruv.shah@ibm.com",
    install_requires=install_requires,
    extras_require={"dev": ["ipython", "black", "pytest", "build", "wheel", "twine"]},
    entry_points={"mlflow.deployments": "watsonml=mlflow_watsonml.deploy"},
    python_requires=">=3.9",
)

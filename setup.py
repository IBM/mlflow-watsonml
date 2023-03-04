from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setup(
    name="mlflow-watsonml",
    version=version,
    description="WatsonML MLflow deployment plugin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/mlflow-watsonml",
    package_dir={"": "mlflow_watsonml"},
    packages=find_packages(where="mlflow_watsonml"),
    author="IBM AI Model Factory team",
    author_email="dhruv.shah@ibm.com",
    install_requires=[
        "mlflow",
        "ibm_watson_machine_learning",
        "python_dotenv",
        "joblib",
    ],
    extra_requires={"dev": ["build", "wheel", "twine"]},
    entry_points={"mlflow.deployments": "watsonml=mlflow_watsonml.deploy"},
    python_requires=">=3.9",
)

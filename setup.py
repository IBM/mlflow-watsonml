from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()


with open("requirements.txt", "r") as fh:
    install_requires = fh.readlines()

exec(open("mlflow_watsonml/_version.py").read())

setup(
    name="mlflow-watsonml",
    version=__version__,  # type: ignore
    description="WatsonML MLflow deployment plugin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/mlflow-watsonml",
    packages=["mlflow_watsonml"],
    author="IBM AI Model Factory team",
    author_email="dhruv.shah@ibm.com",
    install_requires=install_requires,
    extras_require={
        "dev": ["ipython", "black", "pytest", "build", "wheel", "twine", "pytest-cov"],
        "onnx": ["onnx", "onnxruntime"],
        "docs": ["mkdocs", "mkdocstrings-python", "mkdocs-material"],
    },
    entry_points={"mlflow.deployments": "watsonml=mlflow_watsonml.deploy"},
    python_requires=">=3.9",
    # use_scm_version=True,
    # setup_requires=["setuptools_scm"],
)

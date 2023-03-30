# %%[markdown]
"""
The goal of this exercise is to create a WML deployment from mlflow
models with a custom environment specification using conda.yaml

The workflow for this experiment will be as follows - 

1. Create a trained ML model and log it using mlflow (figure out which
flavor to use)
2. Create a WML deployment that takes the model as an input and the 
environment from conda yaml
3. Test the deployment and model scoring
4. Check that it works with local MLflow URI and remote MLflow URI
"""

# %%

import mlflow

# import piputils
from mlflow.deployments import get_deploy_client
from mlflow.models import Model
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from mlflow_watsonml.deploy import WatsonMLDeploymentClient
from mlflow_watsonml.utils import *
from mlflow_watsonml.wml import *

# %%
plugin: WatsonMLDeploymentClient = get_deploy_client("watsonml")

# %%
train = True
iris = load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
if train:
    with mlflow.start_run() as run:
        print(run.info.run_id)

        linear_lr = LogisticRegression()
        linear_lr.fit(X, y)

        mlflow.sklearn.log_model(linear_lr, "model")

# %%
run_id = "24fc7da475c04dda94dbb13d00af18a8"
model_uri = f"runs:/{run_id}/model"
model = Model.load(model_uri)

conda_yaml = mlflow.pyfunc.get_model_dependencies(model_uri, "conda")

deployment_name = "mlflow_watsonml_test_base"

# with open(pip_reqs, "r") as f:
#     requirements = f.read().splitlines()

# %%
client = plugin.get_wml_client()
base_sw_spec_uid = client.software_specifications.get_uid_by_name("runtime-22.2-py3.10")
# %%
meta_prop_pkg_extn = {
    client.package_extensions.ConfigurationMetaNames.NAME: "mlflow conda env",
    client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "MLFlow model env",
    client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml",
}

pkg_extn_details = client.package_extensions.store(
    meta_props=meta_prop_pkg_extn, file_path=conda_yaml
)
pkg_extn_uid = client.package_extensions.get_uid(pkg_extn_details)
pkg_extn_url = client.package_extensions.get_href(pkg_extn_details)

# %%

meta_prop_sw_spec = {
    client.software_specifications.ConfigurationMetaNames.NAME: "mlflow watsonml software_spec",
    client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for mlflow models",
    client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {
        "guid": base_sw_spec_uid
    },
}

sw_spec_details = client.software_specifications.store(meta_props=meta_prop_sw_spec)
sw_spec_uid = client.software_specifications.get_uid(sw_spec_details)

client.software_specifications.add_package_extension(sw_spec_uid, pkg_extn_uid)

# %%
model_details = store_model(
    client,
    model,
    sw_spec_uid,
    deployment_name,
    "mlflow_model",
    "scikit-learn_1.1",
)

# %%
model_id = get_model_id_from_model_details(client=client, model_details=model_details)

# %%
deployment_details = deploy_model(
    client=client,
    name=deployment_name,
    model_id=model_id,
    batch=False,
)

# print(deployment_details)

# %%
print(plugin.predict(deployment_name, X))

# %%

packages = client.software_specifications.get_details(base_sw_spec_uid)["entity"][
    "software_specification"
]["software_configuration"]["included_packages"]

import yaml

# packages = [
#     {'name': 'numpy', 'version': '1.21.2', 'type': 'conda'},
#     {'name': 'pandas', 'version': '1.3.4', 'type': 'pip'},
#     {'name': 'matplotlib', 'version': '3.4.3', 'type': 'conda'}
# ]

conda_packages = []
pip_packages = []

for package in packages:
    if package["type"] == "conda":
        conda_packages.append(f"{package['name']}={package['version']}")
    elif package["type"] == "pip":
        pip_packages.append(f"{package['name']}=={package['version']}")

yaml_data = {
    "name": "wml_runtime-22.2-py3.10",
    "channels": ["defaults", "conda-forge", "anaconda"],
    "dependencies": [{"conda": conda_packages}, {"pip": pip_packages}],
}

with open("conda.yaml", "w") as f:
    yaml.dump(yaml_data, f)

# %%

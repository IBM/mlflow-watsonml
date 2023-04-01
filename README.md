# Mlflow-WatsonML

A plugin that integrates [WatsonML](http://ibm-wml-api-pyclient.mybluemix.net) with MLflow pipeline.
``mlflow_watsonml`` enables mlflow users to deploy mlflow pipeline models into WatsonML.
Command line APIs of the plugin (also accessible through mlflow's python package) makes the deployment process seamless.

## Installation
Plugin package which is available in pypi and can be installed using

```bash
pip install mlflow-watsonml
```

## What does it do
Installing this package uses python's entrypoint mechanism to register the plugin into MLflow's plugin registry. This registry will be invoked each time you launch MLflow script or command line argument.

## Authentication
In order to connect to WatsonML, refer to [.env.template](.env.template)


### Create deployment
The `create` command line argument and ``create_deployment`` python
APIs does the deployment of a model built with MLflow to WatsonML.

##### CLI
```shell script
mlflow deployments create -t watsonml -m <model-uri> --name <deployment-name> -C "software_spec_type=runtime-22.2-py3.10"
```

##### Python API
```python
from mlflow.deployments import get_deploy_client

target_uri = 'watsonml'
plugin = get_deploy_client(target_uri)

plugin.create_deployment(
    name=<deployment-name>, 
    model_uri=<model-uri>, 
    config={"software_spec_type": "runtime-22.2-py3.10"}
)
```

### Update deployment
Update API can used to modify the configuration parameters such as number of workers, version etc., of an already deployed model.
WatsonML will make sure the user experience is seamless while changing the model in a live environment.

##### CLI
```shell script
mlflow deployments update -t watsonml --name <deployment name> -C "software_spec_type=runtime-22.1-py3.10"
```

##### Python API
```python
plugin.update_deployment(name=<deployment name>, config={"software_spec_type": "runtime-22.1-py3.10"})
```

### Delete deployment
Delete an existing deployment. Exception will be raised if the model is not already deployed.

##### CLI
```shell script
mlflow deployments delete -t watsonml --name <deployment name / version number>
```

##### Python API
```python
plugin.delete_deployment(name=<deployment name>)
```

### List all deployments
Lists the names of all the models deployed on the configured WatsonML.

##### CLI
```shell script
mlflow deployments list -t watsonml
```

##### Python API
```python
plugin.list_deployments()
```

### Get deployment details
Get API fetches the details of the deployed model. By default, Get API fetches all the versions of the deployed model.

##### CLI
```shell script
mlflow deployments get -t watsonml --name <deployment name>
```

##### Python API
```python
plugin.get_deployment(name=<deployment name>)
```

### Run Prediction on deployed model
Predict API enables to run prediction on the deployed model.

For the prediction inputs, DataFrame and JSON formats are supported. The python API supports all of these three formats. When invoked via command line, one needs to pass the json file path that contains the inputs.

##### CLI
```shell script
mlflow deployments predict -t watsonml --name <deployment name> --input-path <input file path> --output-path <output file path>
```

output-path is an optional parameter. Without output path parameter result will be printed in console.

##### Python API
```python
plugin.predict(name=<deployment name>, df=<prediction input>)
```

### Plugin help
Run the following command to get the plugin help string.

##### CLI
```shell script
mlflow deployments help -t watsonml
```

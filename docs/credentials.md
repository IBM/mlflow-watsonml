# Getting started

## Setting up Credentials

The best way to setup the credentials is to create a `.env` file and load it as environment variables.

!!! note

    ONLY USE THE CASE THAT APPLIES TO YOU IN YOUR .env FILE
    COPYING ALL CASES CAN OVERWRITE CREDENTIALS AND MAY LEAD TO UNEXPECTED CONNECTION

Refer to the following links for setting up the credentials -

### [Cloud Pak for Data as a Service](https://ibm.github.io/watson-machine-learning-sdk/setup_cloud.html#authentication)

- Case 0: Using a WML Credentials JSON file
```
WML_CREDENTIALS_FILE="/path/to/wml_credentials.json"
```

- Case 1: Using API Key
```
APIKEY="<wml api key>"
LOCATION="<deployment location>"
URL="https://${LOCATION}.ml.cloud.ibm.com"
```

- Case 2: Using Token
```
TOKEN="<wml token>"
LOCATION="<deployment location>"
URL="https://${LOCATION}.ml.cloud.ibm.com"
```

### [Cloud Pak for Data](https://ibm.github.io/watson-machine-learning-sdk/setup_cpd.html#authentication)
- Case 3: Using Username and Password
```
USERNAME="<CP4D Account Username>"
PASSWORD="<CP4D Account Password>"
URL="<CP4D Web URL>"
INSTANCE_ID="openshift"
VERSION="4.6"
```


- Case 4: Using Username and API Key
```
USERNAME="<CP4D Account Username>"
APIKEY="<CP4D Account API Key>"
URL="<CP4D Web URL>"
INSTANCE_ID="openshift"
VERSION="4.6"
```

- Case 5: Using Access Token from CP4D notebook environment
```
TOKEN=${USER_ACCESS_TOKEN}
URL="<CP4D Web URL>"
INSTANCE_ID="openshift"
VERSION="4.6"
```

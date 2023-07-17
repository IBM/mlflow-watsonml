import logging

from ibm_watson_machine_learning.client import APIClient
from ibm_watson_machine_learning.deployments import Deployments
from ibm_watson_machine_learning.platform_spaces import PlatformSpaces
from ibm_watson_machine_learning.repository import Repository
from ibm_watson_machine_learning.Set import Set
from ibm_watson_machine_learning.sw_spec import SwSpec

LOGGER = logging.getLogger(__name__)


class MockAPIClient(APIClient):
    def __init__(self, wml_credentials: dict):
        self.wml_credentials = wml_credentials

        if self.wml_credentials["apikey"] == "incorrect_api_key":
            raise Exception("Connection Failed!")

        self.deployments = MockDeployments(self)
        self.repository = MockRepository(self)
        self.set = MockSet(self)
        self.software_specifications = MockSwSpec(self)
        self.spaces = MockPlatformSpaces(self)


class MockDeployments(Deployments):
    def __init__(self, client):
        self._client = client
        self._deployments = [
            {"metadata": {"name": "deployment_1", "id": "id_of_deployment_1"}},
            {"metadata": {"name": "deployment_2", "id": "id_of_deployment_2"}},
        ]

    @staticmethod
    def get_id(deployment_details):
        return deployment_details["metadata"]["id"]

    def score(self, deployment_id, meta_props, transaction_id=None):
        return {}

    def get_details(
        self,
        deployment_uid=None,
        serving_name=None,
        limit=None,
        asynchronous=False,
        get_all=False,
        spec_state=None,
        _silent=False,
    ):
        if get_all:
            return {"resources": self._deployments}

    def create(self, artifact_uid=None, meta_props=None, rev_id=None, **kwargs):
        return {}

    def update(self, deployment_uid, changes):
        return {}

    def delete(self, deployment_uid):
        return {}


class MockPlatformSpaces(PlatformSpaces):
    def __init__(self, client):
        self._client = client
        self._spaces = [
            {"entity": {"name": "space_1"}, "metadata": {"id": "id_of_space_1"}},
            {"entity": {"name": "space_2"}, "metadata": {"id": "id_of_space_2"}},
        ]

    def get_details(self, space_id=None, limit=None, asynchronous=False, get_all=False):
        if get_all:
            return {"resources": self._spaces}

        space_details = None
        for space in self._spaces:
            if space["metadata"]["id"] == space_id:
                space_details = space
                break

        if space_details is None:
            raise Exception("Invalid space id!")
        else:
            return space_details


class MockRepository(Repository):
    def __init__(self, client):
        self._client = client
        self._models = [
            {"entity": {}, "metadata": {"name": "model_1", "id": "id_of_model_1"}},
            {"entity": {}, "metadata": {"name": "model_2", "id": "id_of_model_2"}},
            {"entity": {}, "metadata": {"name": "model_3", "id": "id_of_model_3"}},
        ]

    def store_model(
        self,
        model,
        meta_props=None,
        training_data=None,
        training_target=None,
        pipeline=None,
        feature_names=None,
        label_column_names=None,
        subtrainingId=None,
        round_number=None,
        experiment_metadata=None,
        training_id=None,
    ):
        return {}

    def update_model(
        self,
        model_uid,
        updated_meta_props=None,
        update_model=None,
    ):
        return {}

    def delete(self, artifact_uid):
        return {}

    def create_model_revision(self, model_uid):
        return {}

    @staticmethod
    def get_model_id(model_details):
        return model_details["metadata"]["id"]

    def list_models(
        self,
        limit=None,
        asynchronous=False,
        get_all=False,
        return_as_df=True,
    ):
        return {}

    def get_model_details(
        self,
        model_uid=None,
        limit=None,
        asynchronous=False,
        get_all=False,
        spec_state=None,
    ):
        if get_all:
            return {"resources": self._models}


class MockSet(Set):
    def __init__(self, client: MockAPIClient):
        self._client = client

    def default_space(self, space_uid):
        for space in self._client.spaces._spaces["resources"]:
            if space["metadata"]["id"] == space_uid:
                return "SUCCESS"

        raise Exception("Invalid Deployment Space!")


class MockSwSpec(SwSpec):
    def __init__(self, client):
        self._client = client
        self._sw_specs = {"space_1": "id_of_space_1", "space_2": "id_of_space_2"}

    def get_id_by_name(self, sw_spec_name):
        return self._sw_specs.get(sw_spec_name, "Not Found")

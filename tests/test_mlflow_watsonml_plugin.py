import os
import shutil
import unittest

import mlflow
from ibm_watson_machine_learning.client import APIClient
from mlflow.deployments import get_deploy_client

from mlflow_watsonml.deploy import WatsonMLDeploymentClient


class TestAssetHealth(unittest.TestCase):
    """Test Asset Health Class"""

    @classmethod
    def setUpClass(cls):
        """Setup method: Called once before test-cases execution"""
        cls.const = 1
        cls.plugin: WatsonMLDeploymentClient = get_deploy_client("watsonml")

    @classmethod
    def tearDownClass(cls):
        """teardown class method: Called once after test-cases execution"""
        ...

    def test_init(self):
        """Test init"""
        test_class = self.__class__

        self.assertEqual(test_class.const, 1)

    def test_mlflow_wml_plugin_import(self):
        """Test if able to get WatsonMLDeployment plugin from mlflow"""

        plugin = self.__class__.plugin
        self.assertIsInstance(plugin, WatsonMLDeploymentClient)
        self.assertIsInstance(plugin.get_wml_client(), APIClient)

    def test_sklearn_wml_store_model(self):
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        from mlflow_watsonml.utils import get_model_id_from_model_details
        from mlflow_watsonml.wml import store_model

        plugin = self.__class__.plugin
        iris = load_iris()
        X = iris.data[:, :2]  # we only take the first two features.
        y = iris.target
        linear_lr = LogisticRegression()
        linear_lr.fit(X, y)

        client = plugin.get_wml_client()

        software_spec_uid = client.software_specifications.get_uid_by_name(
            "runtime-22.1-py3.9"
        )

        model_id = None

        try:
            model_details = store_model(
                client=client,
                model_object=linear_lr,
                software_spec_uid=software_spec_uid,
                name="test",
                model_description="some vague explanation",
                model_type="scikit-learn_1.0",
            )

            model_id = get_model_id_from_model_details(
                client=client, model_details=model_details
            )
            self.assertIsInstance(model_details, dict)
            print(model_details)
            self.assertIsInstance(model_id, str)
            print(model_id)

        finally:
            if model_id is not None:
                client.repository.delete(model_id)

    def test_mlflow_wml_store_model(self):
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        from mlflow_watsonml.utils import get_model_id_from_model_details
        from mlflow_watsonml.wml import store_model

        plugin = self.__class__.plugin
        iris = load_iris()
        X = iris.data[:, :2]  # we only take the first two features.
        y = iris.target
        linear_lr = LogisticRegression()
        linear_lr.fit(X, y)

        model_path = os.path.abspath("./linear_lr")
        if not os.path.exists(model_path):
            mlflow.sklearn.save_model(linear_lr, model_path)

        model_object = mlflow.pyfunc.load_model(model_path)

        client = plugin.get_wml_client()

        software_spec_uid = client.software_specifications.get_uid_by_name(
            "runtime-22.1-py3.9"
        )

        model_id = None

        try:
            model_details = store_model(
                client=client,
                model_object=model_object,
                software_spec_uid=software_spec_uid,
                name="test",
                model_description="some vague explanation",
                model_type="scikit-learn_1.0",
            )

            model_id = get_model_id_from_model_details(
                client=client, model_details=model_details
            )
            self.assertIsInstance(model_details, dict)
            print(model_details)
            self.assertIsInstance(model_id, str)
            print(model_id)

        finally:
            if model_id is not None:
                client.repository.delete(model_id)
            shutil.rmtree(model_path)

    def test_sklearn_wml_deploy_model(self):
        import joblib
        import sklearn
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        self.assertEqual(sklearn.__version__, "1.0.2")
        self.assertEqual(joblib.__version__, "1.1.1")

        from mlflow_watsonml.utils import (
            get_deployment_id_from_deployment_details,
            get_model_id_from_model_details,
        )
        from mlflow_watsonml.wml import deploy_model, store_model

        plugin = self.__class__.plugin
        iris = load_iris()
        X = iris.data[:, :2]  # we only take the first two features.
        y = iris.target
        linear_lr = LogisticRegression()
        linear_lr.fit(X, y)

        client = plugin.get_wml_client()

        software_spec_uid = client.software_specifications.get_uid_by_name(
            "runtime-22.1-py3.9"
        )

        deployment_id = None
        model_id = None

        try:
            model_details = store_model(
                client=client,
                model_object=linear_lr,
                software_spec_uid=software_spec_uid,
                name="test",
                model_description="some vague explanation",
                model_type="scikit-learn_1.0",
            )

            model_id = get_model_id_from_model_details(
                client=client, model_details=model_details
            )

            deployment_details = deploy_model(
                client=client,
                name="test",
                model_id=model_id,
            )

            deployment_id = get_deployment_id_from_deployment_details(
                client=client, deployment_details=deployment_details
            )

            self.assertIsInstance(deployment_details, dict)
            print(deployment_details)
            self.assertIsInstance(deployment_id, str)
            print(deployment_id)

        finally:
            if deployment_id is not None:
                client.deployments.delete(deployment_id)
            if model_id is not None:
                client.repository.delete(model_id)

    def test_mlflow_wml_deploy_model(self):
        import joblib
        import sklearn
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        self.assertEqual(sklearn.__version__, "1.0.2")
        self.assertEqual(joblib.__version__, "1.1.1")

        from mlflow_watsonml.utils import (
            get_deployment_id_from_deployment_details,
            get_model_id_from_model_details,
        )
        from mlflow_watsonml.wml import deploy_model, store_model

        plugin = self.__class__.plugin
        iris = load_iris()
        X = iris.data[:, :2]  # we only take the first two features.
        y = iris.target
        linear_lr = LogisticRegression()
        linear_lr.fit(X, y)

        model_path = os.path.abspath("./linear_lr")
        if not os.path.exists(model_path):
            mlflow.sklearn.save_model(linear_lr, model_path)

        model_object = mlflow.pyfunc.load_model(model_path)

        client = plugin.get_wml_client()

        software_spec_uid = client.software_specifications.get_uid_by_name(
            "runtime-22.1-py3.9"
        )

        deployment_id = None
        model_id = None

        try:
            model_details = store_model(
                client=client,
                model_object=model_object,
                software_spec_uid=software_spec_uid,
                name="test",
                model_description="some vague explanation",
                model_type="scikit-learn_1.0",
            )

            model_id = get_model_id_from_model_details(
                client=client, model_details=model_details
            )

            deployment_details = deploy_model(
                client=client,
                name="test",
                model_id=model_id,
            )

            deployment_id = get_deployment_id_from_deployment_details(
                client=client, deployment_details=deployment_details
            )

            self.assertIsInstance(deployment_details, dict)
            print(deployment_details)
            self.assertIsInstance(deployment_id, str)
            print(deployment_id)

        finally:
            if deployment_id is not None:
                client.deployments.delete(deployment_id)
            if model_id is not None:
                client.repository.delete(model_id)
            shutil.rmtree(model_path)

import unittest

import mlflow
from mlflow.deployments import get_deploy_client


class TestAssetHealth(unittest.TestCase):
    """Test Asset Health Class"""

    @classmethod
    def setUpClass(cls):
        """Setup method: Called once before test-cases execution"""
        cls.const = 1

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
        from ibm_watson_machine_learning.client import APIClient

        from mlflow_watsonml.deploy import WatsonMLDeploymentClient

        plugin = get_deploy_client("watsonml")
        self.assertIsInstance(plugin, WatsonMLDeploymentClient)
        self.assertIsInstance(plugin._wml_client, APIClient)


        

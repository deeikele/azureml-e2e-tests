import unittest
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import json

class TestAzureMLWorkspace(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('config.json') as f:
            config = json.load(f)
        cls.ml_client = MLClient(
            DefaultAzureCredential(),
            config['subscription_id'],
            config['resource_group'],
            config['workspace_name']
        )

    def test_submit_job(self):
        job = self.ml_client.jobs.create_or_update({
            "experiment_name": "test-experiment",
            "command": "echo Hello World"
        })
        self.assertIsNotNone(job)

    def test_submit_pipeline_job(self):
        pipeline_job = self.ml_client.jobs.create_or_update({
            "experiment_name": "test-pipeline",
            "jobs": {
                "step1": {
                    "type": "command",
                    "command": "echo Step 1"
                },
                "step2": {
                    "type": "command",
                    "command": "echo Step 2"
                }
            }
        })
        self.assertIsNotNone(pipeline_job)

    def test_create_custom_environment(self):
        environment = self.ml_client.environments.create_or_update({
            "name": "custom-env",
            "docker": {
                "base_image": "mcr.microsoft.com/azureml/base:latest"
            }
        })
        self.assertIsNotNone(environment)

    def test_create_dataset(self):
        dataset = self.ml_client.data.create_or_update({
            "name": "dummy-dataset",
            "description": "A dummy dataset",
            "data": [
                {"feature1": 1, "feature2": 2},
                {"feature1": 3, "feature2": 4}
            ]
        })
        self.assertIsNotNone(dataset)

    def test_create_compute_instance(self):
        compute_instance = self.ml_client.compute.create_or_update({
            "name": "test-compute",
            "size": "STANDARD_D2_V2"
        })
        self.assertIsNotNone(compute_instance)

    def test_create_component(self):
        component = self.ml_client.components.create_or_update({
            "name": "test-component",
            "type": "command",
            "command": "echo Component"
        })
        self.assertIsNotNone(component)

    def test_create_prompt_flow(self):
        prompt_flow = self.ml_client.prompt_flows.create_or_update({
            "name": "test-prompt-flow",
            "steps": [
                {"type": "prompt", "message": "Hello, what is your name?"},
                {"type": "response", "variable": "name"}
            ]
        })
        self.assertIsNotNone(prompt_flow)

if __name__ == '__main__':
    unittest.main()

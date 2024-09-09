import unittest
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Job,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.ai.ml import command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
from azure.ai.ml.entities import ComputeInstance, AmlCompute
import datetime
import json
import logging

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

    def test_submit_serverless_job(self):

        # define the command
        command_job = command(
            code="./src",
            command="python main.py --iris-csv ${{inputs.iris_csv}} --learning-rate ${{inputs.learning_rate}} --boosting ${{inputs.boosting}}",
            environment="AzureML-lightgbm-3.2-ubuntu18.04-py37-cpu@latest",
            inputs={
                "iris_csv": Input(
                    type="uri_file",
                    path="https://azuremlexamples.blob.core.windows.net/datasets/iris.csv",
                ),
                "learning_rate": 0.9,
                "boosting": "gbdt",
            }
        )

        # submit the command
        job = self.ml_client.jobs.create_or_update(command_job)

        self.assertEqual(job.status, "Starting")

    def test_create_compute_instance(self):
        # Compute Instances need to have a unique name across the region.
        # Here we create a unique name with current datetime
        ci_basic_name = "basic-ci" + datetime.datetime.now().strftime("%Y%m%d%H%M")

        ci_basic = ComputeInstance(name=ci_basic_name, size="STANDARD_DS3_v2")
        returned_ci = self.ml_client.begin_create_or_update(ci_basic).result()
        
        self.assertIsNotNone(returned_ci)


    def test_create_environment(self):

        custom_env_name = "aml-scikit-learn"

        custom_job_env = Environment(
            name=custom_env_name,
            description="Custom environment for Credit Card Defaults job",
            tags={"scikit-learn": "1.0.2"},
            conda_file="environment.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        )
        custom_job_env = self.ml_client.environments.create_or_update(custom_job_env)

        self.assertIsNotNone(custom_job_env)

    def test_create_data_asset(self):
        from azure.ai.ml.entities import Data
        from azure.ai.ml.constants import AssetTypes

        # update the 'my_path' variable to match the location of where you downloaded the data on your
        # local filesystem

        my_path = "./sample_data/default_of_credit_card_clients.csv"
        # set the version number of the data asset
        v1 = "initial"

        my_data = Data(
            name="credit-card",
            version=v1,
            description="Credit card data",
            path=my_path,
            type=AssetTypes.URI_FILE,
        )

        ## create data asset if it doesn't already exist:
        try:
            data_asset = self.ml_client.data.get(name="credit-card", version=v1)
            print(
                f"Data asset already exists. Name: {my_data.name}, version: {my_data.version}"
            )
        except:
            data_asset = self.ml_client.data.create_or_update(my_data)
            print(f"Data asset created. Name: {my_data.name}, version: {my_data.version}")
        self.assertIsNotNone(data_asset)


    def test_register_model(self):
        
        model_name = "model" + datetime.datetime.now().strftime("%m%d%H%M%f")

        model = Model(
            path="./samples_endpoints/online/model-1/model/sklearn_regression_model.pkl",
            name=model_name,
            # type=AssetTypes.CUSTOM_MODEL,
            description="A description of your model",
            tags={"key": "value"}
        )
        
        registered_model = self.ml_client.models.create_or_update(model)
        
        self.assertIsNotNone(registered_model)


    def test_deploy_model(self):
        # Define a random endpoint name
        endpoint_name = "endpt-" + datetime.datetime.now().strftime("%m%d%H%M%f")

        # create an online endpoint
        endpoint = ManagedOnlineEndpoint(
            name = endpoint_name, 
            description="this is a sample endpoint",
            auth_mode="key"
        )

        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        model = Model(path="./samples_endpoints/online/model-1/model/sklearn_regression_model.pkl")
        
        env = Environment(
            conda_file="./samples_endpoints/online/model-1/environment/conda.yaml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        )

        blue_deployment = ManagedOnlineDeployment(
            name="blue",
            endpoint_name=endpoint_name,
            model=model,
            environment=env,
            code_configuration=CodeConfiguration(
                code="./samples_endpoints/online/model-1/onlinescoring",
                scoring_script="score.py"
            ),
            instance_type="Standard_DS3_v2",
            instance_count=1,
        )
        self.ml_client.online_deployments.begin_create_or_update(
            deployment=blue_deployment
        )

        status = self.ml_client.online_endpoints.get(name=endpoint_name).provisioning_state
        
        self.assertEqual(status, "Succeeded")

        # TODO: check for success in deployment creation


    # def test_submit_pipeline_job(self):
    #     pipeline_job = self.ml_client.jobs.create_or_update({
    #         "experiment_name": "test-pipeline",
    #         "jobs": {
    #             "step1": {
    #                 "type": "command",
    #                 "command": "echo Step 1"
    #             },
    #             "step2": {
    #                 "type": "command",
    #                 "command": "echo Step 2"
    #             }
    #         }
    #     })
    #     self.assertIsNotNone(pipeline_job)

    # def test_create_dataset(self):
    #     dataset = self.ml_client.data.create_or_update({
    #         "name": "dummy-dataset",
    #         "description": "A dummy dataset",
    #         "data": [
    #             {"feature1": 1, "feature2": 2},
    #             {"feature1": 3, "feature2": 4}
    #         ]
    #     })
    #     self.assertIsNotNone(dataset)

    # def test_create_component(self):
    #     component = self.ml_client.components.create_or_update({
    #         "name": "test-component",
    #         "type": "command",
    #         "command": "echo Component"
    #     })
    #     self.assertIsNotNone(component)

    # def test_create_prompt_flow(self):
    #     prompt_flow = self.ml_client.prompt_flows.create_or_update({
    #         "name": "test-prompt-flow",
    #         "steps": [
    #             {"type": "prompt", "message": "Hello, what is your name?"},
    #             {"type": "response", "variable": "name"}
    #         ]
    #     })
    #     self.assertIsNotNone(prompt_flow)

if __name__ == '__main__':
    unittest.main()

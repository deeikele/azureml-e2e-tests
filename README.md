# Azure ML End-to-end-tests

Set of tests to run against an Azure Machine Learning workspace and detect any regressions after a backend change.

## Prompt

Generate a number of unit tests in python, to run the following test cases against an Azure Machine Learning workspace. The workspace should be loaded using MLclient and a json config file. Tests: 1. Submit a job 2. Submit a pipeline job 3. Create a cusotm environment 4. Create a dataset with some dummy data 5. create compute instance 6. create a component 7. create a prompt flow
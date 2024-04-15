import argparse
import logging
from datetime import datetime

import httpx
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)
from azure.identity import DefaultAzureCredential

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("azure").setLevel(logging.WARNING)
logger = logging.getLogger("azure-ml-deploy")

# fmt: off
arg_parser = argparse.ArgumentParser(description="Deploy an Azure ML endpoint and deployment.")
arg_parser.add_argument("--subscription-id", type=str, required=True, help="Azure subscription ID")
arg_parser.add_argument("--resource-group", type=str, required=True, help="Azure resource group name")
arg_parser.add_argument("--workspace-name", type=str, required=True, help="Azure ML workspace name")
arg_parser.add_argument("--endpoint-name", type=str, required=True, help="Azure ML endpoint name")
arg_parser.add_argument("--acr-name", type=str, required=True, help="Azure Container Registry name")
arg_parser.add_argument("--image-name", type=str, required=True, help="Docker image name")
arg_parser.add_argument("--image-tag", type=str, required=True, help="Docker image tag")
arg_parser.add_argument("--instance-type", type=str, required=True, help="Azure ML instance type")
# fmt: on

args = arg_parser.parse_args()

# The credential is required
credential = DefaultAzureCredential()

# The MLClient configures Azure ML
ml_client = MLClient(
    credential=credential,
    subscription_id=args.subscription_id,
    resource_group_name=args.resource_group,
    workspace_name=args.workspace_name,
)

# Configure an AML endpoint
endpoint = ManagedOnlineEndpoint(name=args.endpoint_name)

# Create the endpoint
logging.info(f"Creating/Updating endpoint {endpoint.name}")
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Configure a model environment
# This configuration must match with how you set up your API
environment = Environment(
    name=f"{args.image_name}-env",
    image=f"{args.acr_name}.azurecr.io/{args.image_name}:{args.image_tag}",
    inference_config={
        "scoring_route": {
            "port": 8000,
            "path": "/predict",
        },
        "liveness_route": {
            "port": 8000,
            "path": "/health",
        },
        "readiness_route": {
            "port": 8000,
            "path": "/health",  # We use the same as liveness route
        },
    },
)

# Configure the deployment
deployment = ManagedOnlineDeployment(
    name=f"dp-{datetime.now():%y%m%d%H%M%S}",
    endpoint_name=endpoint.name,
    model=None,
    environment=environment,
    instance_type=args.instance_type,
    instance_count=1,  # we only use 1 instance
)

# Create the online deployment.
# Note that this takes approximately 8 to 10 minutes.
# This is a limitation of Azure. We cannot speed it up.
logging.info(f"Creating/Updating deployment {deployment.name}")
deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()


test_data = {"values": [[0, 1], [1, 2]]}

# This assumes we use a key to authenticate to our endpoint
endpoint_token = ml_client.online_endpoints.get_keys(name=endpoint.name).primary_key
headers = {
    "Authorization": f"Bearer {endpoint_token}",
    "Content-Type": "application/json",
    "azureml-model-deployment": deployment.name,
}

# Send a request to the endpoint to test it
logging.info(f"Testing endpoint {endpoint.name} at {endpoint.scoring_uri}")
response = httpx.post(endpoint.scoring_uri, json=test_data, headers=headers)
try:
    response.raise_for_status()
except Exception:
    # When our test fails, we delete the deployment and stop the program
    logger.info(f"Endpoint response error {response.status_code}: {response.text}")
    logs = ml_client.online_deployments.get_logs(
        name=deployment.name, endpoint_name=endpoint.name, lines=50
    )
    logger.info(f"Deployment logs: \n{logs}")

    logger.info(f"Removing failed deployment {deployment.name}")
    ml_client.online_deployments.begin_delete(
        name=deployment.name, endpoint_name=endpoint.name
    ).result()

    raise SystemExit("Deployment failed.")

# When everything goes well, we change all endpoint traffic to the new deployment
logging.info(f"Switching 100% traffic to deployment {endpoint.name}")
endpoint.traffic = {deployment.name: 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Delete all old deployments
logging.info("Removing older deployments")
for existing_deployment in ml_client.online_deployments.list(
    endpoint_name=endpoint.name
):
    if existing_deployment.name != deployment.name:
        ml_client.online_deployments.begin_delete(
            endpoint_name=endpoint.name, name=existing_deployment.name
        ).result()

# from inference_sdk import InferenceHTTPClient

# client = InferenceHTTPClient(
#     api_url="http://localhost:9001",
#     api_key="ZTyY4JmpoYwoYcSrVRzk"
# )

# # Run the workflow
# result = client.infer(
#     workspace_name="yash-sonar-dui7q",  # from your Roboflow URL
#     workflow_id="vision-guard-wovga--2",  # project-name--version-number
#     images={
#         "image": "J:/projects Yash python/Vision Guard/test_cap.jpg"
#     }
# )

# print(result)
# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ZTyY4JmpoYwoYcSrVRzk"
)

# infer on a local image
result = CLIENT.infer("J:/projects Yash python/Vision Guard/test_cap.jpg", model_id="vision-guard-wovga/2")
print(result)

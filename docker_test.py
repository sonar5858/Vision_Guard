# from inference_sdk import InferenceHTTPClient

# client = InferenceHTTPClient(
#     api_url="http://localhost:9001",  # Docker server
#     api_key="
# )

# result = client.run_workflow(
#     workspace_name="yash-sonar-dui7q",
#     workflow_id="detect-count-and-visualize-2",
#     images={
#         "image": "J:/projects Yash python/Vision Guard/test_cap.jpg"
#     }
# )

# print(result)
# from inference_sdk import InferenceHTTPClient

# client = InferenceHTTPClient(
#     api_url="http://localhost:9001", # use local inference server
#     api_key="
# )

# result = client.run_workflow(
#     workspace_name="yash-sonar-dui7q",
#     workflow_id="yashmodel",
#     images={
#         "image": r"J:\projects Yash python\Vision Guard\test_cap.jpg"
#     }    
# )
# Import the InferencePipeline object
from inference import InferencePipeline
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Display predictions using matplotlib (instead of cv2.imshow)
def my_sink(result, video_frame):
    if result.get("output_image"):
        image_np = result["output_image"].numpy_image  # Convert to numpy array

        plt.imshow(image_np)
        plt.title("Workflow Detection Frame")
        plt.axis("off")
        plt.pause(0.001)  # Short pause to update plot

        plt.clf()  # Clear figure for next frame
    print(result)

# Initialize the pipeline
pipeline = InferencePipeline.init_with_workflow(
    api_key=os.getenv("roboflow_api_key"),
    workspace_name="yash-sonar-dui7q",
    workflow_id="yashmodel",  # Use your actual workflow ID
    video_reference=0,  # Webcam input
    max_fps=10,  # Lowered for stability
    on_prediction=my_sink
)

# Start processing
plt.figure()  # Create one persistent window
pipeline.start()
pipeline.join()

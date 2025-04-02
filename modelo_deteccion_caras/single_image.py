import requests
import json
import base64
from PIL import Image, ImageDraw
import io
import numpy as np
import matplotlib.pyplot as plt
import sys

# Function to load and encode image


def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()

    base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_encoded}"


def main():
    # Get image path from command line argument or use default
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"

    try:
        # Encode image
        encoded_image = encode_image(image_path)

        # Format for MLServer inference API
        inference_request = {
            "inputs": [
                {
                    "name": "images",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [encoded_image]
                }
            ]
        }

        # Send the request
        url = "http://localhost:8080/v2/models/faces-detection/infer"
        response = requests.post(url, json=inference_request)

        # Print status code
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            # Parse response
            result = response.json()
            print(result)
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

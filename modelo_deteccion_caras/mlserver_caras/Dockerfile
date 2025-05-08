FROM python:3.9-slim

WORKDIR /app

# Install required packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget

# Install Python dependencies
RUN pip install mlserver opencv-python-headless numpy Pillow scikit-learn

# Create directory structure
RUN mkdir -p models/faces-detection

# Copy model files
COPY custom_model.py .
COPY model-settings.json ./models/faces-detection/
COPY settings.json .
COPY haarcascade_frontalface_default.xml ./models/faces-detection/
COPY haarcascade_frontalface_default.xml .


# Expose ports
EXPOSE 8080
EXPOSE 8081
EXPOSE 8082

# Start MLServer
CMD mlserver start .

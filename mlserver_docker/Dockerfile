FROM python:3.9-slim

WORKDIR /app

# Install required packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget

# Install Python dependencies
RUN pip install mlserver opencv-python-headless numpy Pillow

# Create directory structure
RUN mkdir -p models/object-detection

# Copy model files
COPY yolo_detector.py .
COPY custom_model.py .
COPY model-settings.json ./models/object-detection/
COPY settings.json .

# Download required model files
RUN wget -O yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights && \
    wget -O yolov3-tiny.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg && \
    wget -O coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names && \
    mv yolov3-tiny.weights yolov3-tiny.cfg coco.names ./models/object-detection/

# Expose ports
EXPOSE 8080
EXPOSE 8081
EXPOSE 8082

# Start MLServer
CMD mlserver start .
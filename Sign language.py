import tensorflow as tf
import numpy as np
import cv2  # Import OpenCV
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = "saved_model"  # Adjust the path accordingly
LABEL_MAP_PATH = "label_map.pbtxt"  # Adjust the path accordingly
IP_CAMERA_URL = "http://192.168./video"  # Replace with your IP camera URL

print('Loading model...', end='')
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

# Load the label map
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

def detect_and_visualize(frame):
    image_np = np.array(frame)

    # The input needs to be a tensor, convert it using tf.convert_to_tensor.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with tf.newaxis.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    # All outputs are batch tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    # Visualization
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.4,  # Adjust this value to set the minimum probability boxes to be classified as True
        agnostic_mode=False
    )

    return image_np_with_detections

# Initialize the video capture
cap = cv2.VideoCapture(IP_CAMERA_URL)

if not cap.isOpened():
    print("Error: Could not open video stream from IP camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform detection and visualization
    frame_with_detections = detect_and_visualize(frame)

    # Display the resulting frame
    cv2.imshow('Detection Result', frame_with_detections)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
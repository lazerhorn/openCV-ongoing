import numpy as np
import cv2

image_path = 'pictures/th (8).jpeg'  # Use forward slashes or raw string literals for file paths
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'

min_confidence = 0.2

classes = ['person']
np.random.seed(54321)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Read the original image
image = cv2.imread(image_path)

# Get the shape of the original image
height, width = image.shape[0], image.shape[1]

# Convert the image into a binary large object
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)

net.setInput(blob)
detected_objects = net.forward()

# Create a copy of the original image
image_copy = image.copy()

for i in range(detected_objects.shape[2]):
    confidence = detected_objects[0, 0, i, 2]
    if confidence > min_confidence:
        class_index = int(detected_objects[0, 0, i, 1])
        upper_left_x = int(detected_objects[0, 0, i, 3] * width)
        upper_left_y = int(detected_objects[0, 0, i, 4] * height)
        lower_left_x = int(detected_objects[0, 0, i, 5] * width)
        lower_left_y = int(detected_objects[0, 0, i, 6] * height)

        prediction_text = f"{classes[0]}: {confidence:.2f}"
        cv2.rectangle(image_copy, (upper_left_x, upper_left_y), (lower_left_x, lower_left_y), colors[0], 3)
        cv2.putText(image_copy, prediction_text,
                    (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                    cv2.FONT_HERSHEY_PLAIN, 0.6, colors[0], 2)

# Display the modified copy of the image with original size
cv2.imshow('detected objects', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

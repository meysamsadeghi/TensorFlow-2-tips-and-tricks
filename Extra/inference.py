import time
import tensorflow as tf
from leran.images.image_processing import (load_image_pixels,
                                           decode_netout,
                                           correct_yolo_boxes,
                                           do_nms,
                                           get_boxes,
                                           draw_boxes,
                                           load_img_for_yolov3_input)


# define the labels
LABELS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

INTERESTING_OBJECTS_COLOR_DICT = {"person": 'r', "chair": 'm', "cell phone": 'g', "laptop": 'y',
                                  "bicycle": 'k', "car": 'b', "backpack": 'm', "apple": 'c',
                                  "orange": 'k', "book": 'y', "keyboard": 'm'}
OTHER_OBJECTS_COLOR_DICT = {k: 'w' for k in LABELS if k not in INTERESTING_OBJECTS_COLOR_DICT.keys()}
COLOR_DICT = {**INTERESTING_OBJECTS_COLOR_DICT, **OTHER_OBJECTS_COLOR_DICT}

IMAGE_SIZE = 416 # Same height and width are used as input dimenssion for network


# define the expected input shape for the model
input_w, input_h = IMAGE_SIZE, IMAGE_SIZE


# load yolov3 model
model = tf.keras.models.load_model('./yoloV3_TF2/yolov3_model.h5')

process_time = time.time()
# load and prepare image
PHOTO_PATH = './images/jpg_three.jpg'
image, image_w, image_h = load_img_for_yolov3_input(PHOTO_PATH, IMAGE_SIZE, IMAGE_SIZE)
# make prediction
yhat = model.predict(image)
# summarize the shape of the list of arrays
print([a.shape for a in yhat])
# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
# define the probability threshold for detected objects
class_threshold = 0.6
boxes = list()
for i in range(len(yhat)):
	# decode the output of the network
	boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
# correct the sizes of the bounding boxes for the shape of the image
correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
# suppress non-maximal boxes
do_nms(boxes, 0.5)

# get the details of the detected objects
v_boxes, v_labels, v_scores = get_boxes(boxes, LABELS, class_threshold)
# summarize what we found
for i in range(len(v_boxes)):
	print(v_labels[i], v_scores[i])
# draw what we found
draw_boxes(PHOTO_PATH, v_boxes, v_labels, v_scores, COLOR_DICT)

process_time = time.time() - process_time
print(process_time)

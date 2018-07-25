#
# Author: roshan 
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.


## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb



# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_faces'
VIDEO_NAME = 'video.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()


#TO Do. Auto input all files in directory using a loop
#input 12 emojis 
mypath  =  CWD_PATH + "/EMOJI/" 
#onlyfiles = [ f for f in os.listdir(mypath)]
dab = cv2.imread(os.path.join(mypath,'dab.jpg'),cv2.IMREAD_COLOR)
happy = cv2.imread(os.path.join(mypath,'happy.jpg'),cv2.IMREAD_COLOR)
hug = cv2.imread(os.path.join(mypath,'hug.jpg'),cv2.IMREAD_COLOR)
idc = cv2.imread(os.path.join(mypath,'idc.jpg'),cv2.IMREAD_COLOR)
innocent = cv2.imread(os.path.join(mypath,'innocent.jpg'),cv2.IMREAD_COLOR)
kiss = cv2.imread(os.path.join(mypath,'kiss.jpg'),cv2.IMREAD_COLOR)
smile = cv2.imread(os.path.join(mypath,'smile.jpg'),cv2.IMREAD_COLOR)
thinking = cv2.imread(os.path.join(mypath,'thinking.jpg'),cv2.IMREAD_COLOR)
toungue = cv2.imread(os.path.join(mypath,'toungue.jpg'),cv2.IMREAD_COLOR)
veryhappy = cv2.imread(os.path.join(mypath,'veryhappy.jpg'),cv2.IMREAD_COLOR)
wink = cv2.imread(os.path.join(mypath,'wink.jpg'),cv2.IMREAD_COLOR)
wtf = cv2.imread(os.path.join(mypath,'wtf.jpg'),cv2.IMREAD_COLOR)


#function to return a CV2 frame, given an emoji name and a detection frame

def getemo (hh,frame):
    if hh != [] :
        if hh[0]["name"] == "dab":
            z= overlay(frame,dab)
          
        elif hh[0]["name"] == "happy":
            z= overlay(frame,happy)
          

        elif hh[0]["name"] == "hug" :
            z= overlay(frame,hug)
           

        elif hh[0]["name"] == "idc" :
            z= overlay(frame,idc)
           

        elif hh[0]["name"] == "innocent" :
            z= overlay(frame,innocent)
           

        elif hh[0]["name"] == "kiss" :
            z= overlay(frame,kiss)
           

        elif hh[0]["name"] == "smile" :
            z= overlay(frame,smile)
            

        elif hh[0]["name"] == "thinking" :
            z= overlay(frame,thinking)
           

        elif hh[0]["name"] == "toungue" :
            z= overlay(frame,toungue)
           

        elif hh[0]["name"] == "veryhappy" :
            z= overlay(frame,veryhappy)
            

        elif hh[0]["name"] == "wink" :
            z= overlay(frame,wink)
           
        else :
            z= overlay(frame,wtf)
           

    else:
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        fontcolor = (0, 0, 255)
        z = cv2.putText(frame,'No Emotions Detected',(450,130), fontface, fontscale, fontcolor) 
    return(z)


#images = np.empty(len(onlyfiles), dtype=object)
#for n in range(0, len(onlyfiles)):
#  images[n] = cv2.imread( os.path.join(mypath,onlyfiles[n]),-1)


#Placing a smiley on an image at a given piont
def overlay (frame,mark):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    mark = cv2.cvtColor(mark, cv2.COLOR_BGR2BGRA)
    frame_h, frame_w, frame_c = frame.shape
    mark_h, mark_w, mark_c = mark.shape
    for i in range(0, mark_h):
        for j in range(0,mark_w):
            frame[150 + i,950 + j] = mark[i,j]
    return(frame)


# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','faces-detection.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 12

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

k=0
while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    #print([category_index.get(i) for i in classes[0]])


    op = [category_index.get(value) for (index,value) in enumerate(classes[0]) if scores[0,index] > 0.8]
    
    # Draw the results of the detection (aka 'visulaize the results')
    z1 = vis_util.visualize_boxes_and_labels_on_image_array(frame,
        np.squeeze(boxes), np.squeeze(classes).astype(np.int32), 
        np.squeeze(scores), category_index, use_normalized_coordinates=True,
        line_thickness=8, min_score_thresh=0.80)

# All the results have been drawn on the frame, so it's time to display it.
    #cv2.imshow('Object detector', frame)
    x = getemo(op, z1)
    cv2.imshow('Object detector', x)
    cv2.imwrite("gz/img%d.jpg" %k, x)
    k += 1
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

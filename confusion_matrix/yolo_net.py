import numpy as np
import cv2
from skimage import io
import os

class Yolo():
	'Yolo-network i.e give a image input and it gives back a object detection'

	def net(self,image,file_path,weights_name,cfg_name,par_confidence=0.25,threshold =0.4):

		weightsPath = os.path.sep.join([file_path, weights_name])
		configPath = os.path.sep.join([file_path, cfg_name])

		net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

		image = io.imread(image)
		(H, W) = image.shape[:2]

		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608,608),swapRB=True, crop=False)
		net.setInput(blob)
		layerOutputs = net.forward(ln)


		boxes = []
		confidences = []
		classIDs = []

		for output in layerOutputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				if confidence > par_confidence:

					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		idxs = cv2.dnn.NMSBoxes(boxes, confidences, par_confidence,threshold)

		box_list        =[]
		class_list      =[]
		confidence_list =[]
		for k in idxs:
			box_list        += [boxes[k[0]]]
			class_list      += [classIDs[k[0]]]
			confidence_list += [confidences[k[0]]]

		return {'boxes':box_list,'class_Ids':class_list,'confidences':confidence_list}

#image ='/home/Dados/Geovista/darknetcolab/buracos/1140-2-VID_20200514_124351975.jpg'
#obj= Yolo()
#print(obj.net(image,'cfg/'))

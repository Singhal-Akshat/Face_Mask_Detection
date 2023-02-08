from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

def detect_and_predict_mask(frame, faceNet, maskNet):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	# mean value of red, green and blue
	faceNet.setInput(blob)
	detections = faceNet.forward() # getting details of different faces in the frame
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5: # a confidence threshold
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			#valid dimensions
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
            #gathering all the faces in a single frame all at once place
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	if len(faces) > 0:
		#for faster prediction we perform prediction of mask on every face in the frame at the same time
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	#return face locations for making a rectangle and mask or not mask label
	return (locs, preds)

print("[INFO] loading face detector model...")
prototxtPath = "Model\\deploy.prototxt"
weightsPath = "Model\\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("Model\\mask_detector.model")

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	#getting all faces with corresponding label from frame
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame)
	# key = cv2.waitKey(1) & 0xFF
	key = cv2.waitKey(1)
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()

    
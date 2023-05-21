
import base64
from deepface.detectors import FaceDetector
import cv2
import numpy as np


def loadBase64Img(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

# Using Retina face
def extract_face(image, resize=(224, 224)):
	detector_backend = 'retinaface'
	face_detector = FaceDetector.build_model(detector_backend)
	faces = FaceDetector.detect_faces(face_detector, detector_backend, image, align = False)
	facesInb64=[]
	for face, (x, y, w, h) in faces:
		if w > 130: #discard small detected faces
			face_boundary = image[int(y):int(y+h), int(x):int(x+w)]
			face_image=cv2.resize(face_boundary,resize)
			res, frame = cv2.imencode('.jpg', face_image)   
			b64 = base64.b64encode(frame) 
			img = "data:image/jpeg;base64," + b64.decode('utf-8')
			facesInb64.append(img)
	return facesInb64
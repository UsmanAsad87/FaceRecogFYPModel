
import base64
import datetime
import os
import pickle
import uuid
from api_helper.classes import TimeStamp, User
from api_helper.helper_func import extract_face, loadBase64Img
from api_helper.mongoDbFunc import getAllUser
from flask import Flask, jsonify, request, make_response,render_template,redirect
import pymongo
import pandas as pd
import matplotlib.image
import pandas as pd
from deepface import DeepFace



myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["FaceRecog"]
collection=mydb["Users"]


def addPerson(img,personName):
	if img.filename=='':
		return render_template('add_person.html', alert_message="Image has no file name.")

	filename=img.filename.split(".")
	name=filename[0]
	ext=filename[1]
	img_bytes = img.read()
	encoded_string = base64.b64encode(img_bytes)
	str_encoded_string=str(encoded_string)
	str_encoded_string=str_encoded_string[2:]
	instance='data:image/'+ext+';base64,'+str_encoded_string
	instance=instance[:-1]
	resultDf=pd.DataFrame()
	try:
		resultDf = DeepFace.find(instance
		, db_path = 'dataset_small'
		, model_name = 'ArcFace'
		, distance_metric = 'cosine'
		, detector_backend = 'retinaface'
		, silent=True
	)
	except Exception as err:
		return render_template('add_person.html', alert_message="Some error occured.")
	
	
	if resultDf.empty:
		return addPersonHelper(img=instance,personName=personName)
	else:
		for index, row in resultDf.iterrows():
			if row["ArcFace_cosine"] < 0.56:
				return render_template('add_person.html', alert_message="Person Already exist.")

		return addPersonHelper(img=instance,personName=personName)

def addPersonHelper(img,personName):

	model_name = "ArcFace"; distance_metric = "cosine"; detector_backend = 'retinaface'
	resultDf=pd.DataFrame()
	img2=loadBase64Img(img)
	resp_all={}
	face_imgs=extract_face(img2)

	if(len(face_imgs)>1):
		return render_template('add_person.html', alert_message="Image consist multiple faces kindly upload another image.")
	if(len(face_imgs)==0):
		return render_template('add_person.html', alert_message="Image has no face of person.")

	for face_img in face_imgs:
		try:	
			faceImg = DeepFace.detectFace(img_path = face_img, target_size=(224, 224), enforce_detection = False, detector_backend = 'retinaface', align = True)
			count = uuid.uuid1()
			newpath = 'dataset_small/'+str(count)  
			if not os.path.exists(newpath):
				os.makedirs(newpath)

			save_path='dataset_small/'+str(count)+'/image'+str(count)+'.png'
			matplotlib.image.imsave(save_path, faceImg)
			#for updating the embeddings
			file_name="representations_arcface.pkl"
			db_path='dataset_small'
			f = open(db_path+'/'+file_name, 'rb')
			representations = pickle.load(f)
			rep= DeepFace.represent(save_path,model_name="ArcFace",detector_backend = 'retinaface')
			instance=[]
			instance.append(save_path)
			instance.append(rep)
			representations.append(instance)
			f = open(db_path+'/'+file_name, "wb")
			pickle.dump(representations, f)
			f.close()
			ID=save_path.split("/")


			with open(save_path, "rb") as img_file:
				my_string = base64.b64encode(img_file.read())

			rec={"name":personName,"id":ID[1],"imgUrl":my_string.decode("utf-8"),"recent_timeStamp":datetime.min,'recent_location':'none',"timeStamps":[]}
			collection.insert_one(rec)


		except Exception as err:
			return render_template('add_person.html', alert_message="Some error occured while adding person.")

	
	return render_template('add_person.html', success_message="Person added successfully.")



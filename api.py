import shutil
import warnings
from api_helper.helper_func import extract_face, loadBase64Img
from api_helper.mongoDbFunc import addTimeStampOfUser, getAllUser, resetMongoDb
from api_helper.search_person import searchByImg, searchByName


warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#------------------------------

from flask import Flask, jsonify, request, make_response,render_template,redirect
from api_helper.classes import TimeStamp, User
import argparse
import uuid
import json
import time
from tqdm import tqdm
import matplotlib.image
import pandas as pd
import pickle
from deepface.commons import functions
from deepface.detectors import FaceDetector
import pymongo
import base64
import numpy as np
from datetime import datetime
import uuid
import cv2



from flask import redirect,url_for ,render_template



#------------------------------

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

#------------------------------

if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)

#------------------------------

from deepface import DeepFace

#------------------------------

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate memory on the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU memory growth enabled")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


app = Flask(__name__)
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["FaceRecog"]
collection=mydb["Users"]

#------------------------------

if tf_version == 1:
	graph = tf.get_default_graph()

#------------------------------
#Service API Interface
@app.route('/',methods=['GET','POST'])
def view():
	#Search User
	if request.method=='POST':
		#If it is search by name
		if request.form.get('name') != None:
			name=request.form['name']
			return searchByName(name=name)

		#else search By image
		searchImg = request.files['newcomer']
		return searchByImg(img=searchImg)


	#If you want to get all users	
	return getAllUser()


@app.route('/details/<string:id>',methods=['GET','POST'])
def details(id):
	user=collection.find_one({"id":id})
	stamps=[]
	for item in user['timeStamps']:
		stamp=TimeStamp(item)
		stamps.append(stamp)

	stamps=sorted(stamps,key= lambda x: x.time,reverse=True)
	userData= User(user,stamps)
	return render_template('details.html',user=userData)

@app.route('/delete/<string:id>',methods=['GET','POST'])
def delete(id):
	save_path = 'dataset_small/' + id + '/image' + id + '.png'
	folder_path= 'dataset_small/' + id 
	file_name = "representations_arcface.pkl"
	db_path = 'dataset_small'

	# Load existing representations
	with open(os.path.join(db_path, file_name), 'rb') as f:
		representations = pickle.load(f)

	# Find and remove the instance with the specified save_path
	for instance in representations:
		if instance[0] == save_path:
			representations.remove(instance)
			break

	# Save updated representations
	with open(os.path.join(db_path, file_name), 'wb') as f:
		pickle.dump(representations, f)

	collection.delete_one({"id":id})

	if os.path.exists(folder_path):
		shutil.rmtree(folder_path)

	return redirect('/')


@app.route('/deleteStamp/<string:id>/<string:time>', methods=['GET', 'POST'])
def deleteStamp(id, time):
	print(id)
	print(time)
	# one=collection.find_one({"id":ID[1]})
	# # print(one)
	# timeStamps=one["timeStamps"]
	# now=datetime.now()
	# #location="lab"
	# # print(now)
	
	# timeStamp={"time":now,"location":location,'img':img}
	# timeStamps.append(timeStamp)
	# # print(timeStamp)
	# prev={"id":ID[1]}
	# nextt={"$set":{"timeStamps":timeStamps,"recent_timeStamp":now,'recent_location':location}}
	# up=collection.update_many(prev,nextt)
	# print(up.modified_count)

	# collection.delete_one({"id":id})

	return redirect('/details/'+id)


@app.route('/add_person',methods=['GET','POST'])
def addNewPerson():
	if request.method == 'POST':
		newcomer = request.files.get('newcomer')
		name = request.form.get('name')
		if not newcomer or not name:
			return render_template('add_person.html', alert_message="Please provide the image and name of person.")
		
		return addPerson(img=newcomer,personName=name)
		
	return render_template('add_person.html')


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




@app.route('/update/<string:id>',methods=['GET','POST'])
def update(id):
	user=collection.find_one({"id":id})
	stamps=[]
	for item in user['timeStamps']:
		stamp=TimeStamp(item)
		stamps.append(stamp)
	userData= User(user,stamps)

	if request.method == 'POST':
		name = request.form.get('name')

		if  not name:
			return render_template('update.html',user=userData, alert_message="Please provide updated name of person.")
		
		prev={"id":id}
		nextt={"$set":{"name":name}}
		up=collection.update_many(prev,nextt)
		print(up.modified_count)
		return render_template('update.html',user=userData, alert_message="Name updated successfully.")
		
	return render_template('update.html',user=userData)




@app.route('/findface', methods=['POST'])
def findface():

	global graph

	tic = time.time()
	req = request.get_json()
	trx_id = uuid.uuid4()

	resp_obj = jsonify({'success': False})

	if tf_version == 1:
		with graph.as_default():
			resp_obj = findfaceWrapper(req, trx_id)
	elif tf_version == 2:
		resp_obj = findfaceWrapper(req, trx_id)

	#--------------------------

	toc =  time.time()
	print("TOTAL TIME:"+str(toc-tic))

	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic

	return resp_obj, 200

@app.route('/test')
def test():
	resp_obj = jsonify({'success': True})
	return resp_obj, 200

def findfaceWrapper(req, trx_id = 0):

	resp_obj = jsonify({'success': False})

	#-------------------------------------
	#find out model

	model_name = "ArcFace"; distance_metric = "cosine"; detector_backend = 'retinaface'; location='unknown'

	if "model_name" in list(req.keys()):
		model_name = req["model_name"]
	
	if "location" in list(req.keys()):
		location= req["location"]

	if "detector_backend" in list(req.keys()):
		detector_backend = req["detector_backend"]

	#-------------------------------------
	#retrieve images from request

	img = ""
	if "img" in list(req.keys()):
		img = req["img"] #list
		#print("img: ", img)

	validate_img = False
	if len(img) > 11 and img[0:11] == "data:image/":
		validate_img = True

	if validate_img != True:
		print("invalid image passed!")
		return jsonify({'success': False, 'error': 'you must pass img as base64 encoded string'}), 205

	resultDf=pd.DataFrame()

	#Just to check

	tic1 =  time.time()
	img2=loadBase64Img(img)
	toc1 =  time.time()
	print("loadBase64Img TIME:"+str(toc1-tic1))
	resp_all={}
	
	tic2 = time.time()
	face_imgs=extract_face(img2)
	toc2 = time.time()
	print("extract_faces TIME:"+str(toc2-tic2))

	for face_img in face_imgs:
		
		try:
			tic3 =  time.time()

			resultDf = DeepFace.find(face_img
				, db_path = "dataset_small"
				, model_name = model_name
				, distance_metric = distance_metric
				, detector_backend = detector_backend
				, silent=True
			)

			toc3 =  time.time()
			print("FIND_FACE TIME:"+str(toc3-tic3))

		except Exception as err:
			print("Exception: ",str(err))
			resp_obj = jsonify({'success': False, 'error': str(err)}), 205

		#-------------------------------------

		tic4 =  time.time()
		resp_obj = {}
		if not resultDf.empty:
			# print(resultDf)
			resp_obj['face_found']= 'True'
			topMatchDf=resultDf.nsmallest(1, 'ArcFace_cosine')
			imgurl=topMatchDf['identity'][0]
			print(topMatchDf['ArcFace_cosine'][0])
			if(topMatchDf['ArcFace_cosine'][0] <0.56):
				resp_obj['imgurl']= imgurl
				addTimeStampOfUser(imgurl=imgurl,location=location,img=face_img)
		

		toc4 =  time.time()
		print("POST Processing TIME:"+str(toc4-tic4))

		if resultDf.empty or topMatchDf['ArcFace_cosine'][0] >0.56:
			resp_obj['face_found']= 'False'
			resp_obj['imgurl']= 'None'
			try:	
				# face = DeepFace.detectFace2(face_img)
				face=True
				if(face):
					resp_obj['HasFace']= face
					resp_obj['face_found']= 'true'
					faceImg = DeepFace.detectFace(img_path = face_img, target_size=(224, 224), enforce_detection = False, detector_backend = 'retinaface', align = True)
					count = uuid.uuid1()
					print(count)
					newpath = 'dataset_small/'+str(count)  
					if not os.path.exists(newpath):
						os.makedirs(newpath)

					save_path='dataset_small/'+str(count)+'/image'+str(count)+'.png'
					resp_obj['imgurl']= save_path
					matplotlib.image.imsave(save_path, faceImg)
					resp_obj['faceAdded']= 'true'

					#for updating the embeddings
					file_name="representations_arcface.pkl"
					db_path='dataset_small'
					f = open(db_path+'/'+file_name, 'rb')
					representations = pickle.load(f)
					# img_path="dataset_small/10/image10.png"
					rep= DeepFace.represent(save_path,model_name="ArcFace",detector_backend = 'retinaface')
					instance=[]
					instance.append(save_path)
					instance.append(rep)
					representations.append(instance)
					f = open(db_path+'/'+file_name, "wb")
					pickle.dump(representations, f)
					f.close()
					ID=save_path.split("/")
					print(ID[1])
					name = ID[1].split("-")[0]
					
					now=datetime.now()
					#location="lab"
					timeStamp={"time":now,"location":location,"img":face_img}

					with open(save_path, "rb") as img_file:
						my_string = base64.b64encode(img_file.read())

					rec={"name":name,"id":ID[1],"imgUrl":my_string.decode("utf-8"),"timeStamps":[timeStamp,],'recent_timeStamp':now,'recent_location':location}
					collection.insert_one(rec)

				else:
					resp_obj['HasFace']= face

			except Exception as err:
				print("Exception: ",str(err))
				resp_obj = jsonify({'success': False, 'error': str(err)}), 205
				return resp_obj

		# resp_obj["resultDf"] = resultDf.to_json()
		resp_all['resp'+str(len(face_img))]=resp_obj

	if(len(face_imgs)==0):
		resp_all['face_found']= 'False'
		# print("usman")
	
	return resp_all





if __name__ == '__main__':
	# resetMongoDb()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-p', '--port',
		type=int,
		default=5000,
		help='Port of serving api')
	args = parser.parse_args()

	#app.run(host='0.0.0.0', port=80,debug=False)
	# app.run(host='0.0.0.0', port=args.port,debug=True,threaded=True)
	# app.run(host='10.25.28.172', port=5000,debug=False)  #'192.168.0.106' home
	app.run(host='0.0.0.0', port=args.port,debug=False,threaded=True)
	# app.run(host='0.0.0.0', port=args.port,debug=True)

	# app.run( port=args.port,debug=True)





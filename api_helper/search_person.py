import base64
from api_helper.classes import TimeStamp, User
from api_helper.mongoDbFunc import getAllUser
from flask import Flask, jsonify, request, make_response,render_template,redirect
import pymongo
import pandas as pd

from deepface import DeepFace



myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["FaceRecog"]
collection=mydb["Users"]

def searchByImg(img):
	if img.filename=='':
		return getAllUser()

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
		resp_obj = jsonify({'success': False, 'error': str(err)}), 205
	
	if resultDf.empty:
		print("Not found")
		return render_template('index.html',allTodo=[])
	else:
		print(resultDf)
		Docs=[]	
		IDs=[]
		for index, row in resultDf.iterrows():
			imgurl=row['identity']
			imgurl= imgurl.replace("\\", "/")
			id=imgurl.split("/")[1]
			if row["ArcFace_cosine"] < 0.56 and not id in IDs:
				imgurl=row['identity']
				user=collection.find_one({"name":id})
				IDs.append(id)
		
				#if no user is found then this will run
				if(user==None):
					print("none")
					return render_template('index.html',allTodo=Docs)

				stamps=[]
				for item in user['timeStamps']:
					stamp=TimeStamp(item)
					stamps.append(stamp)

				userData= User(user,stamps)
				Docs.append(userData)



       
		print(Docs)
		return render_template('index.html',allTodo=Docs)




def searchByName(name):

	#if search query is empty
	if(name==''):
		return getAllUser()
	
	#if no user is found then this will run
	user=collection.find_one({"name":name})
	if(user==None):
		return render_template('index.html',allTodo=[])

	#if user record is found	
	Docs=[]	
	stamps=[]
	for item in user['timeStamps']:
		stamp=TimeStamp(item)
		stamps.append(stamp)
		print("DATA: "+stamp.location+" "+stamp.time+"  ")
	userData= User(user,stamps)
	Docs.append(userData)
	for st in userData.timeStamps:
		print(st.time)
	return render_template('index.html',allTodo=Docs)



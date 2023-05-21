import base64
from datetime import datetime
import os
from classes import TimeStamp, User
from flask import Flask, jsonify, request, make_response,render_template,redirect
import pymongo



myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["FaceRecog"]
collection=mydb["Users"]

def getAllUser():
	allDocs=collection.find({}).sort('recent_timeStamp',-1)
	Docs=[]
	for item in allDocs:
		stamps=[]
		for st in item['timeStamps']:
			stamp=TimeStamp(st)
			stamps.append(stamp)
		doc=User(item,stamps)
		Docs.append(doc)
	return render_template('index.html',allTodo=Docs,pic=doc.imgUrl)


def addTimeStampOfUser(imgurl,location,img):
	imgurl= imgurl.replace("\\", "/")
	ID=imgurl.split("/")
	#print(imgurl)
	#print(ID[1])
	print(ID[1])
	one=collection.find_one({"id":ID[1]})
	# print(one)
	timeStamps=one["timeStamps"]
	now=datetime.now()
	#location="lab"
	# print(now)
	
	timeStamp={"time":now,"location":location,'img':img}
	timeStamps.append(timeStamp)
	# print(timeStamp)
	prev={"id":ID[1]}
	nextt={"$set":{"timeStamps":timeStamps,"recent_timeStamp":now,'recent_location':location}}
	up=collection.update_many(prev,nextt)
	print(up.modified_count)
	
def addAllUserInDb(path):
	dir_list = os.listdir(path)
	for item in dir_list:
		if os.path.isdir(os.path.join(path, item)):
			filename=os.listdir(os.path.join(path, item))
			imgpath=os.path.join(path, item,filename[0])
			with open(imgpath, "rb") as img_file:
				my_string = base64.b64encode(img_file.read())
			rec={"name":item.split("-")[0],"id":item,"imgUrl":my_string.decode("utf-8"),"recent_timeStamp":datetime.min,'recent_location':'none',"timeStamps":[]}
			collection.insert_one(rec)

def resetMongoDb():
	# dictionary={"name":"usman","marks":20}
	# collection.insert_one(dictionary)
	collection.delete_many({})
	addAllUserInDb('dataset_small')
	# collection.delete_one({"name":"ID13"})


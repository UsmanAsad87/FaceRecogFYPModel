from datetime import datetime


class User:
	def __init__(self, id,name,recentTime,recentLocation, timeStamps, imgUrl):
		self.id = id
		self.name = name
		self.recentLocation = recentLocation
		self.timeStamps= timeStamps
		self.imgUrl = imgUrl
		self.recentTime = recentTime

	def __init__(self, item):
		self.id = item['id']
		self.name = item['name']
		self.recentLocation ='' if "none" in item['recent_location'] else item['recent_location']
		self.timeStamps= item['timeStamps']
		self.imgUrl ="data:image/png;base64,"+ item["imgUrl"]
		self.recentTime = '' if item['recent_timeStamp']==datetime.min else item['recent_timeStamp'].strftime("%m/%d/%Y, %H:%M:%S")

	def __init__(self, item,stamps):
		self.id = item['id']
		self.name = item['name']
		self.recentLocation ='' if "none" in item['recent_location'] else item['recent_location']
		self.timeStamps= stamps
		self.imgUrl ="data:image/png;base64,"+ item["imgUrl"]
		self.recentTime = '' if item['recent_timeStamp']==datetime.min else item['recent_timeStamp'].strftime("%m/%d/%Y, %H:%M:%S")

class TimeStamp:
	def __init__(self,time,location,img):
		self.time = time
		self.location= location
		self.img = img

	def __init__(self, stamp):
		self.time = stamp['time'].strftime("%m/%d/%Y, %H:%M:%S")
		self.location= stamp['location']
		self.img = stamp["img"]



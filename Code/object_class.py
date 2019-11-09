"""
File: object_class.py
Authors:
Purpose:
Implementation Details:
{"description": null, "alphanumeric_color": "n/a", "alphanumeric": "n/a", "shape": "n/a", "user": "arizona", "background_color": "n/a", "orientation": null, "longitude": null, "latitude": null, "autonomous": true, "type": "standard"}
"""


import json


class TrivialObj:

	def __init__(self):

		self.description = None
		self.user = "AZ"
		self.autonomous = True
		self.type = "standard"

		self.alphanumeric_color = None
		self.alphanumeric = None
		self.shape = None
		self.shape_color = None
		self.alphanumeric_orientation = None

		self.latitude = 0
		self.longitude = 0

	def create_dict(self):
		
		obj_dict = {

			"description" : self.description,
			"alphanumeric_color" : self.alphanumeric_color,
			"alphanumeric" : self.alphanumeric,
			"shape" : self.shape,
			"user" : self.user,
			"background_color" : self.background_color,
			"orientation" : self.alphanumeric_orientation,
			"longitude" : self.longitude,
			"latitude" : self.latitude,
			"autonomous" : self.autonomous,
			"type" : self.type

		}

		return obj_dict


	def create_json(self, target_dir):
		pass

	def init_from_json(self, path_with_file):
		
		loaded_obj_dict = json.load(path_with_file)

		self.description = loaded_obj_dict["description"]
		self.user = loaded_obj_dict["user"]
		self.autonomous = loaded_obj_dict["autonomous"]
		self.type = loaded_obj_dict["type"]

		self.alphanumeric_color = loaded_obj_dict["alphanumeric_color"]
		self.alphanumeric = loaded_obj_dict["alphanumeric"]
		self.shape = loaded_obj_dict["shape"]
		self.shape_color = loaded_obj_dict["shape_color"]
		self.alphanumeric_orientation = loaded_obj_dict["alphanumeric_orientation"]

		self.latitude = loaded_obj_dict["latitude"]
		self.longitude = loaded_obj_dict["longitude"]

	def update_


def main():
	pass


if __name__ == "__main__":
	main()
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


def main():
	pass


if __name__ == "__main__":
	main()
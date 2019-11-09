# classify_images.py
# Author: Arizona Autonomous Vehicles Club
# Task: AUVSI SUAS 2017 Image Classification
# Description: This script is the primary program for competition time.
#			  Once run, it will loop forever until terminated manually,
#			  e.g. with Ctrl+C or Ctrl+Z. The script continuously polls
#			  its current directory (or optional provided directory) for
#			  images of .jpg (or optional specified format) and classify
#			  the image(s). Results will then be transmitted to the
#			  interop server for scoring

import argparse
import os
import sys
import json

import numpy as np
import cv2
import tensorflow as tf

from interop import AsyncClient
from interop import Client
from interop.types import Target
from interop import InteropError

import convnets.wideresnet.wideresnet_model as wideresnet
import mlp.mlp_model as mlp
import shallow_mlp.shallow_mlp_model as shallow_mlp

# Constants
LOGGING = True
DEBUG = False
VERBOSE = True
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
MAX_BATCH_SIZE = 128

# Utility function
def degToOrientation(degrees):
	''' Converts degrees of deviation (from north) to cardinal orientation.
		Args:
			degrees: float in range [-180, 180], where +90 is 90 degrees CCW
		Returns:
			orientation: string indicating orientation, where 90 degrees CCW is NW
	'''
	# Normalize to [0, 360]
	while degrees < 0:
		degrees += 360
	while degrees > 360:
		degrees -= 360	
	if (degrees < 22.5):
		orientation = "N"
	elif (degrees < 67.5):
		orientation = "NW"
	elif (degrees < 112.5):
		orientation = "W"
	elif (degrees < 157.5):
		orientation = "SW"
	elif (degrees < 202.5):
		orientation = "S"
	elif (degrees < 247.5):
		orientation = "SE"
	elif (degrees < 292.5):
		orientation = "E"
	elif (degrees < 337.5):
		orientation = "NE"
	else:
		orientation = "N"
	
	return orientation

class TargetClassifier():
	def __init__(self, userid, password, checkpoint_dir, server_url=None):
		# Store Look Up Tables
		self.shapes = {0 : 'n/a', 1 : 'circle', 2 : 'cross', 3 : 'heptagon', 4 : 'hexagon', 5 : 'octagon', 6 : 'pentagon', 7 : 'quarter_circle', 8 : 'rectangle', 9 : 'semicircle', 10 : 'square', 11 : 'star', 12 : 'trapezoid', 13 : 'triangle'}
		self.alphanums = {0 : 'n/a',  1 : 'A',  2 : 'B',  3 : 'C',  4 : 'D',  5 : 'E',  6 : 'F',  7 : 'G',  8 : 'H',  9 : 'I',  10 : 'J', 11 : 'K', 12 : 'L', 13 : 'M', 14 : 'N', 15 : 'O', 16 : 'P', 17 : 'Q', 18 : 'R', 19 : 'S', 20 : 'T', 21 : 'U', 22 : 'V', 23 : 'W', 24 : 'X', 25 : 'Y', 26 : 'Z', 27 : '0', 28 : '1', 29 : '2', 30 : '3', 31 : '4', 32 : '5', 33 : '6', 34 : '7', 35 : '8', 36 : '9'}
		self.colors = {0 : 'n/a', 1 : 'white', 2 : 'black', 3 : 'gray', 4 : 'red', 5 : 'blue', 6 : 'green', 7 : 'yellow', 8 : 'purple', 9 : 'brown', 10 : 'orange'}

		# Store userid
		self.userid = userid

		# IMPORTANT! Put updated mean standard values here
		self.mean = np.array([83.745, 100.718, 115.504]) # R, G, B
		self.stddev = np.array([53.764, 52.350, 59.265]) # R, G, B
		
		# Counters/trackers for interop
		self.target_id = 2 # Start at target #2

		# Interoperability client
		if server_url is not None:
			self.interop = Client(server_url, userid, password)
		else:
			self.interop = None
			print('Warning: No interop server connection')

		# Logging mode
		if LOGGING:
			self.logging_counter = 0

		# Build TensorFlow graphs
		assert os.path.isdir(checkpoint_dir)
		# Shape graph
		self.shape_graph = tf.Graph()
		with self.shape_graph.as_default():
			self.inputs_shape = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
			self.logits_shape = wideresnet.inference(self.inputs_shape, 14, scope='shapes') # 13 shapes + background
			variable_averages = tf.train.ExponentialMovingAverage(
									wideresnet.MOVING_AVERAGE_DECAY)
			variables_to_restore = variable_averages.variables_to_restore()
			saver = tf.train.Saver(variables_to_restore)
			
			self.shape_sess = tf.Session() # graph=self.shape_graph
			#shape_saver = tf.train.Saver()
			shape_ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint_dir, 'shape'))
			if shape_ckpt and shape_ckpt.model_checkpoint_path:
				print('Reading shape model parameters from %s' % shape_ckpt.model_checkpoint_path)
				#shape_saver.restore(self.shape_sess, self.shape_ckpt.model_checkpoint_path)
				saver.restore(self.shape_sess, shape_ckpt.model_checkpoint_path)
			else:
				print('Error restoring parameters for shape. Ensure checkpoint is stored in ${checkpoint_dir}/shape/')
				# sys.exit(1)
	
		# Shape color graph
		self.shape_color_graph = tf.Graph()
		with self.shape_color_graph.as_default():
			self.inputs_shape_color = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
			self.logits_shape_color = shallow_mlp.inference(self.inputs_shape_color, 11, scope='shape_color') # 10 shape_colors + background
			variable_averages = tf.train.ExponentialMovingAverage(
								shallow_mlp.MOVING_AVERAGE_DECAY)
			variables_to_restore = variable_averages.variables_to_restore()
			saver = tf.train.Saver(variables_to_restore)
			
			self.shape_color_sess = tf.Session() # graph=self.shape_color_graph
			#shape_color_saver = tf.train.Saver()
			shape_color_ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint_dir, 'shape_color'))
			if shape_color_ckpt and shape_color_ckpt.model_checkpoint_path:
				print('Reading shape_color model parameters from %s' % shape_color_ckpt.model_checkpoint_path)
				#shape_color_saver.restore(self.shape_color_sess, self.shape_color_ckpt.model_checkpoint_path)
				saver.restore(self.shape_color_sess, shape_color_ckpt.model_checkpoint_path)
			else:
				print('Error restoring parameters for shape_color. Ensure checkpoint is stored in ${checkpoint_dir}/shape_color/')
				# sys.exit(1)
	
		# Alphanum graph
		self.alphanum_graph = tf.Graph()
		with self.alphanum_graph.as_default():
			self.inputs_alphanum = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
			self.logits_alphanum = wideresnet.inference(self.inputs_alphanum, 37, scope='alphanums') # 37 alphanums + background
			variable_averages = tf.train.ExponentialMovingAverage(
									wideresnet.MOVING_AVERAGE_DECAY)
			variables_to_restore = variable_averages.variables_to_restore()
			saver = tf.train.Saver(variables_to_restore)
			
			self.alphanum_sess = tf.Session()
			#alphanum_saver = tf.train.Saver()
			alphanum_ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint_dir, 'alphanum'))
			if alphanum_ckpt and alphanum_ckpt.model_checkpoint_path:
				print('Reading alphanum model parameters from %s' % alphanum_ckpt.model_checkpoint_path)
				#alphanum_saver.restore(self.alphanum_sess, self.alphanum_ckpt.model_checkpoint_path)
				saver.restore(self.alphanum_sess, alphanum_ckpt.model_checkpoint_path)
			else:
				print('Error restoring parameters for alphanum. Ensure checkpoint is stored in ${checkpoint_dir}/alphanum/')
				# sys.exit(1)
	
		# Alphanum color graph
		self.alphanum_color_graph = tf.Graph()
		with self.alphanum_color_graph.as_default():
			self.inputs_alphanum_color = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
			self.logits_alphanum_color = mlp.inference(self.inputs_alphanum_color, 11, scope='letter_color') # 10 alphanum_colors + background
			variable_averages = tf.train.ExponentialMovingAverage(
									mlp.MOVING_AVERAGE_DECAY)
			variables_to_restore = variable_averages.variables_to_restore()
			saver = tf.train.Saver(variables_to_restore)
			
			self.alphanum_color_sess = tf.Session()
			#alphanum_color_saver = tf.train.Saver()
			alphanum_color_ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint_dir, 'alphanum_color'))
			if alphanum_color_ckpt and alphanum_color_ckpt.model_checkpoint_path:
				print('Reading alphanum_color model parameters from %s' % alphanum_color_ckpt.model_checkpoint_path)
				#alphanum_color_saver.restore(self.alphanum_color_sess, self.alphanum_color_ckpt.model_checkpoint_path)
				saver.restore(self.alphanum_color_sess, alphanum_color_ckpt.model_checkpoint_path)
			else:
				print('Error restoring parameters for alphanum_color. Ensure checkpoint is stored in ${checkpoint_dir}/alphanum_color/')
				# sys.exit(1)

	def preprocess_image(self, image):
		''' Preprocess image for classification
			Args:
				image: np.array containing raw input image
			Returns:
				image: np.array of size [1, width, height, depth]
		'''
		im = image.copy()

		# Change from BGR (OpenCV) to RGB
		b = im[:,:,0].copy()
		im[:,:,0] = im[:,:,2] # Put red channel in [:,:,0]
		im[:,:,2] = b # Put blue channel in [:,:,2]

		# Resize image as necessary
		if (np.greater(im.shape[:2], [IMAGE_SIZE, IMAGE_SIZE]).any()):
			# Scale down
			im = cv2.resize(im, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
		elif (np.less(im.shape[:2], [IMAGE_SIZE, IMAGE_SIZE]).any()):
			# Scale up
			im = cv2.resize(im, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

		# MeanStd normalization
		im = np.subtract(im, self.mean)
		im = np.divide(im, self.stddev)
		# Pad dimensions from 3-D to 4-D if necessary
		if len(im.shape) == 3:
			im = np.expand_dims(im, axis=0)
		return im

	def preprocess_image_hsv(self, image):
		''' Preprocess image for classification
			Args:
				image: np.array containing raw input image
			Returns:
				image: np.array of size [1, width, height, depth]
		'''
		im = image.copy()

		# Resize image as necessary
		if (np.greater(im.shape[:2], [IMAGE_SIZE, IMAGE_SIZE]).any()):
			# Scale down
			im = cv2.resize(im, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
		elif (np.less(im.shape[:2], [IMAGE_SIZE, IMAGE_SIZE]).any()):
			# Scale up
			im = cv2.resize(im, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

		# Change from BGR to HSV
		im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
		im = im.astype(np.float32)

		# Scale to [-1,1]
		im[:,:,0] = np.subtract(im[:,:,0], 89.5) # Hue
		im[:,:,0] = np.divide(im[:,:,0], 89.5) # Hue
		im[:,:,1] = np.subtract(im[:,:,1], 127.5) # Saturation
		im[:,:,1] = np.divide(im[:,:,1], 127.5) # Saturation
		im[:,:,2] = np.subtract(im[:,:,2], 127.5) # Value
		im[:,:,2] = np.divide(im[:,:,2], 127.5) # Value

		if len(im.shape) == 3:
			im = np.expand_dims(im, axis=0)

		return im

	def classify_shape(self, image):
		''' Extract the shape of the target
				Args: The preprocessed input image, of shape 
			Returns:
				str: The classified shape, in human readable text
		'''
		try:
			predictions = self.shape_sess.run([self.logits_shape],
			                                  feed_dict={self.inputs_shape: image})
			class_out = np.argmax(predictions)
			confidence = np.max(predictions)
			if confidence >= 0.50:
				return self.shapes[class_out]
			else:
				print('Shape %s rejected at confidence %f' % (self.shapes[class_out], confidence))
				return None
		# If checkpoint not loaded, ignore error and return None
		except tf.errors.FailedPreconditionError:
			return None
	
	def classify_shape_color(self, image):
		''' Extract the shape color of the target
				Args: The input image
				Returns:
					str: The classified color, in human readable text
		'''
		try:
			predictions = self.shape_color_sess.run([self.logits_shape_color],
				                                 feed_dict={self.inputs_shape_color: image})
			class_out = np.argmax(predictions)
			confidence = np.max(predictions)
			if confidence >= 0.50:
				return self.colors[class_out]
			else:
				print('Shape color %s rejected at confidence %f' % (self.colors[class_out], confidence))
				return None
		# If checkpoint not loaded, ignore error and return None
		except tf.errors.FailedPreconditionError:
			return None
	
	def classify_letter(self, image):
		''' Extract the letter color of the target
				Args: The input image
				Returns: 
					str: The classified letter, in human readable text
					str: Amount rotated clockwise, in degrees (int)
		'''
		try:
			rot = 0
			class_out_dict = {}
			image = image.copy().squeeze()
			(h, w) = image.shape[:2]
			center = (w / 2, h / 2)
			while (rot < 360):
				# Rotate image clockwise by rot degrees
				M = cv2.getRotationMatrix2D(center, rot, 1.0)
				image_rot = cv2.warpAffine(image, M, (w, h))
				image_rot = np.expand_dims(image_rot, axis=0)
				predictions = self.alphanum_sess.run([self.logits_alphanum],
				                                feed_dict={self.inputs_alphanum: image_rot})
				class_out_dict[np.max(predictions)] = (np.argmax(predictions), rot) # TODO: Handle duplicate confidences
				rot += 22.5 # 45 degree stride. If computation budget allows, consider increasing to 22.5 deg
			confidence = max(class_out_dict) # Maximum confidence from classifications
			#class_out = np.argmax(predictions)
			class_out, rot_out = class_out_dict[confidence]
			if confidence >= 0.50:
				return self.alphanums[class_out], rot_out
			else:
				print('Letter %s rejected at confidence %f' % (self.alphanums[class_out], confidence))
				return None, None
		# If checkpoint not loaded, ignore error and return None
		except tf.errors.FailedPreconditionError:
			return None, None
	
	def classify_letter_color(self, image):
		''' Extract the letter color of the target
				Args: The input image
				Returns:
					str: The classified color, in human readable text
		'''
		try:
			predictions = self.alphanum_color_sess.run([self.logits_alphanum_color],
			                                     feed_dict={self.inputs_alphanum_color: image})
			class_out = np.argmax(predictions)
			confidence = np.max(predictions)
			if confidence >= 0.50:
				return self.colors[class_out]
			else:
				print('Letter color %s rejected at confidence %f' % (self.colors[class_out], confidence))
				return None
		# If checkpoint not loaded, ignore error and return None
		except tf.errors.FailedPreconditionError:
			return None

	def check_valid(self, packet):
		''' Check whether the prepared output packet is valid
				Args:
					dict: dictionary (JSON) of proposed output packet
				Returns:
					bool: True if packet is valid, False if not
		'''
		# FIXME: Delete this part
		labels = ["shape", "alphanumeric", "backgorund_color", "alphanumeric_color"]
		for key, value in packet.iteritems():
			# Background class, flagged "n/a" in our translation key
			#if (value == "n/a") and key != "description":
			#	return False
			if (value != "n/a" and value != None) and key in labels:
				print(value)
				return True
			# Background and alphanumeric color should never be the same
			#if packet['background_color'] == packet['alphanumeric_color']:
			#	return False
			# TODO: Check for valid lat/lon
	
		#return True
		return False

	def check_duplicate(self, target):
		''' Utility function to check if target has already been submitted
			Args:
				target: Target to check
			Returns:
				retval: bool, True if duplicate exists
		'''
		if not self.interop:
			return None
		targetLocation = (target.latitude, target.longitude)
		targets = self.interop.get_targets()
		for t in targets:
			tLocation = (t.latitude, t.longitude)
			if self.calc_distance(targetLocation, tLocation) < 0.00015:	
				return True
		return False

	def _calc_distance(self, a, b):
		''' Utility function to calculate the distance between two arrays
			Args:
				a: an array of numbers
				b: an array of numberes
			Returns:
				distance: absolute Euclidian distance between a and b
		'''
		a = np.array(a)
		b = np.array(b)
		assert(a.shape == b.shape)
		return np.sqrt(np.sum(np.power(np.subtract(a, b), 2)))
	
	def classify_and_maybe_transmit(self, image, location, orientation):
		''' Main worker function for image classification. Transmits depending on validity
			Args:
				image: np.array of size [width, height, depth]
				location: tuple of GPS coordinates as (lat, lon)
				orientation: degree value in range [-180, 180],
							 where 0 represents due north and 90 represents due east
		'''
		if image is None:
			return False

		image_orig = image

		image = self.preprocess_image(image)
		imageHSV = self.preprocess_image_hsv(image_orig)

		# Run respective image classifiers
		shape = self.classify_shape(image)
		background_color = self.classify_shape_color(imageHSV)
		alphanumeric, rot = self.classify_letter(image)
		alphanumeric_color = self.classify_letter_color(imageHSV)
		latitude, longitude = location

		# Debugging only
		if DEBUG and orientation is None:
			orientation = 0
		if DEBUG and (latitude, longitude) == (None, None):
			latitude, longitude = (0, 0)

		# Extract orientation
		if orientation is not None and rot is not None:
			orientation += rot
			orientation = degToOrientation(orientation)
		else:
			orientation = None

		if DEBUG or VERBOSE:
			print 'Shape =', shape
			print 'Shape Color =', background_color
			print 'Alphanumeric =', alphanumeric
			print 'Alphanum Color =', alphanumeric_color
			print 'Lat, Lon =', latitude, ',', longitude
			print 'Orientation = ', orientation
	
		packet = {
				"user": self.userid,
				"type": "standard",
				"latitude": latitude,
				"longitude": longitude,
				"orientation": orientation,
				"shape": shape,
				"background_color": background_color,
				"alphanumeric": alphanumeric,
				"alphanumeric_color": alphanumeric_color,
				"description": None,
				"autonomous": True
			}

		if LOGGING:
			if not os.path.exists('processed'):
				os.mkdir('processed')
			savepath = 'processed/img_' + str(self.logging_counter)
			with open(savepath + '.json', 'w') as outfile:
				json.dump(packet, outfile)
			cv2.imwrite(savepath + '.jpg', image_orig)
			self.logging_counter += 1

		# Check for false positives or otherwise invalid targets
		packet_valid = self.check_valid(packet)
		if packet_valid:
			packet["id"] = self.target_id
			json_packet = json.dumps(packet)
			if self.interop is not None:
				if not os.path.exists('transmit'):
					os.mkdir('transmit')
				savepath = 'transmit/img_' + str(self.target_id)
				with open(savepath + '.json', 'w') as outfile:
					json.dump(packet, outfile)
				cv2.imwrite(savepath + '.jpg', image_orig)
				# Transmit data to interop server
				target = Target(id=self.target_id,
								user=self.userid,
								type='standard',
								latitude=latitude,
								longitude=longitude,
								orientation=orientation,
								shape=shape,
								background_color=background_color,
								alphanumeric=alphanumeric,
								alphanumeric_color=alphanumeric_color,
								description=None, autonomous=True)
				if not self.check_duplicate(target):
					try:
						print('Transmitting target %d info' % self.target_id)
						self.interop.post_target(target)
					except Exception as e:
						print(e)
					# Transmit image to interop server
					with open('transmit/img_%d.jpg' % self.target_id) as f:
						im_data = f.read()
						try:
							print('Transmitting target %d image' % self.target_id)
							self.interop.post_target_image(self.target_id, im_data)
						except Exception as e:
							print(e)
				else:
					print('INFO: Duplicate target detected at (%f, %f) lat/lon'
							% (latitude, longitude))
	
				# TODO (optional): build database of detected targets, correct mistakes
			self.target_id += 1
		else:
			print('INFO: An invalid target was discarded')
		return packet_valid, packet

def get_unique_path(rootdir, subdir, filename):
	counter = 0
	processed_dir = subdir + str(counter).zfill(2)
	# Increment counter until we find unused processed_##/file location
	while os.path.exists(os.path.join(rootdir, processed_dir, filename)):
		counter += 1
		processed_dir = subdir + str(counter).zfill(2)
	return processed_dir


def main():
	# Create command line args
	parser = argparse.ArgumentParser(
						description='This program is to be run on the ground station '
									'side of the 2016-17 computer vision system. It'
									'continuously scans a directory for images and passes'
									'them to image classifier(s). Results are sent to the'
									'Interop Server')
	parser.add_argument('-u', '--userid', default='arizona',
							help='User ID for Interop Server.')
	parser.add_argument('-p', '--password', default='3768877561',
							help='Password for Interop Server.')
	parser.add_argument('-s', '--server_url', #default='http://10.10.130.10:80',
							help='URL for Interop Server.')
	parser.add_argument('-f', '--format', default='jpg', 
							help='Input image format. Suggested formats are jpg or png')
	parser.add_argument('-d', '--dir', 
							help='Directory to scan for images. If no directory provided, '
									'scans current working directory')
	parser.add_argument('-c', '--checkpoint_dir', required=True, 
								help='Path to checkpoint directories. '
								'Each classifier should be kept in a separate directory '
								'according to their name (e.g. scope). For example, '
								'checkpoints/ with subdirectories shape/, alphanum/, etc')
	parser.add_argument('-b', '--batch_mode', action='store_true',
								help='Enable batch mode. Batch size is hardcoded (128)')	
	
	args = parser.parse_args()

	# Process command line args
	if args.dir is not None:
		 directory = args.dir
	else:
		 directory = os.getcwd()
	ext = '.' + args.format.split('.')[-1].lower()

	# Validate arguments
	assert os.path.exists(directory)

	# Initialize classifiers
	classifier = TargetClassifier(args.userid, args.password, args.checkpoint_dir, args.server_url)

	print('Running on directory: %s\t\t' % directory)
	print('Searching for images of format: %s\t' % ext)

	print("INFO: Beginning infinite loop. To terminate, use Ctrl+C")
	while True:
		if args.batch_mode:
			batch_counter = 0
			images = []
		# Iterate through files in directory (NOTE: 'file' is a __builtin__)
		for f in os.listdir(directory):
			if f.lower().endswith(ext):

				#Reads in specific image based on type given
				image = cv2.imread(os.path.join(directory, f))
				print(f)

				# Grabs the corresponding number X_img.ext  (ext<-ending type)
				splitFile = f.split("_")
				print(splitFile)
				imageJsonName = os.path.join(directory, splitFile[0]+"_data.json")
				print(imageJsonName)

				#Reads in the data from the JSON file, such as longitude, latitude, direction (<-format in preprocessing script)
				jsonData = None
				try:
					with open(imageJsonName) as json_file:
						jsonData = json.load(json_file)

				except():
					print("No corresponding image Json with captured plane data")

				if args.batch_mode and batch_counter < MAX_BATCH_SIZE:
					images.append(image)
					batch_counter += 1
				else:
					if args.batch_mode:
						image = np.array(images)
						images = []
						batch_counter = 0

					# Code for classifying images and sending, if no Json file exits it simply passes None for latitude, longitude and orientation
					if(jsonData == None):
						packet_valid, labels = classifier.classify_and_maybe_transmit(image, (None,None), None)
					else:
						packet_valid, labels = classifier.classify_and_maybe_transmit(image, (jsonData['latitude'], jsonData['longitude']), jsonData['heading'])

				# Move processed image into processed_## subdir
				if packet_valid:
					processed_dir = get_unique_path(directory, 'processed_', f)

				else:
					processed_dir = get_unique_path(directory, 'falsepositives_', f)

				# NOTE: Program will continue to work after counter > 99, but naming
				#       convention will be violated (e.g. processed_101/foo.jpg)
				# Make subdirectories as necessary
				if not os.path.exists(processed_dir):
					os.mkdir(processed_dir)

				# Move processed file to processed_##/ subdirectory
				os.rename(os.path.join(directory, f), os.path.join(processed_dir, f))
				os.rename(os.path.join(directory, imageJsonName), os.path.join(processed_dir, imageJsonName))

				# Write labels as JSON
				json_filename = f.rsplit('.', 1)[0] + '.json'
				json_filepath = os.path.join(processed_dir, json_filename)
				with open(json_filepath, 'w') as json_file:
					json.dump(labels, json_file)

if __name__ == "__main__":
	print("Welcome to AZA's Image Classification Program")
	print("For options and more information, please rerun this program with the -h option")
	main()


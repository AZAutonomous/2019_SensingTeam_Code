import cv2
import numpy as np
import math
import os
import argparse
import random
import json

from components import *

DEBUG = False

parser = argparse.ArgumentParser(description='Generate sample targets based on resources as specified in components.py')
parser.add_argument('-s', '--size', metavar='size', type=int, help="Output image resolution. Output is always square")
parser.add_argument('format', choices=['json', 'jpg', 'all'], help="Output image format. Acceptable options are json and jpg and all", metavar='format')

args = parser.parse_args()

def randomAngle(maxangle=180):
	return (random.random() - 0.5) * 2 * maxangle
	
def degToOrientation(degrees):
	''' Converts degrees of deviation (from north) to cardinal orientation.
		Args:
			degrees: float in range [-180, 180], where +90 is 90 degrees CCW
		Returns:
			orientation: string indicating orientation, where 90 degrees CCW is NW
	'''
	assert abs(degrees) <= 180
	rot = degrees + 180. # Convert to [0, 360] range
	if (rot < 22.5):
		orientation = "S"
	elif (rot < 67.5):
		orientation = "SE"
	elif (rot < 112.5):
		orientation = "E"
	elif (rot < 157.5):
		orientation = "NE"
	elif (rot < 202.5):
		orientation = "N"
	elif (rot < 247.5):
		orientation = "NW"
	elif (rot < 292.5):
		orientation = "W"
	elif (rot < 337.5):
		orientation = "SW"
	else:
		orientation = "S"
	
	return orientation

def main():
	# Preload backgrounds
	bgs = []
	for background in backgrounds:
		bgs.append(cv2.imread(background[1]))

	# Iterate through shapes
	for shape in shapes:
		# Read shape in. Note current iteration has shape as black on white
		targetShape = cv2.imread(shape[1])
		# Iterate through letters
		for letter in letters:
			# Read letter in. Note current iteration has letter as white on black
			targetLetter = cv2.imread(letter[1])
			# Iterate through target colors
			for color in colors:
				targetColor = color[1]
				# Iterate through letter colors
				for letterColor in colors:
					# Check for same colors, skip (assume will never have)
					if letterColor == color:
						continue
					targetLetterColor = letterColor[1]
					
					# Grab random slice of random background
					bg = bgs[random.randint(0, len(bgs))-1]
					x = random.randint(0, bg.shape[0]-targetShape.shape[0])
					y = random.randint(0, bg.shape[1]-targetShape.shape[1])
					targetBg = bg[x:x+targetShape.shape[0], y:y+targetShape.shape[1]]
					
					# Randomly resize shape and letter
					factor = 1 + 0.2 * random.random()
					newsize = np.multiply(targetShape.shape[:2], factor).astype(np.int32)
					newsize = tuple(newsize)
					sizediff = np.subtract(newsize, targetShape.shape[:2]).astype(np.float32)
					sizediff = np.divide(sizediff, 2)
					targetShapeResized = cv2.resize(targetShape, newsize)
					targetLetterResized = cv2.resize(targetLetter, newsize)
					minX = int(math.floor(sizediff[0]))
					maxX = int(targetShapeResized.shape[0]-math.ceil(sizediff[0]))
					minY = int(math.floor(sizediff[1]))
					maxY = int(targetShapeResized.shape[1]-math.ceil(sizediff[1]))
					targetShapeResized = targetShapeResized[minX:maxX, minY:maxY]
					targetLetterResized = targetLetterResized[minX:maxX, minY:maxY]

					# Randomly rotate shape and letter (by same amount)
					(h, w) = targetShapeResized.shape[:2]
					assert (h,w) == targetLetterResized.shape[:2]
					center = (w / 2, h / 2)
					rot = randomAngle(180)
					orientation = degToOrientation(rot) # Store for labeling later
					M = cv2.getRotationMatrix2D(center, rot, 1.0)
					# Invert images. This is important for proper background padding
					targetShape_gray = cv2.cvtColor(targetShapeResized, cv2.COLOR_BGR2GRAY)
					targetLetter_gray = cv2.cvtColor(targetLetterResized, cv2.COLOR_BGR2GRAY)
					targetShape_gray = (255-targetShape_gray)
					targetLetter_gray = (255-targetLetter_gray)
					targetShape_rot = cv2.warpAffine(targetShape_gray, M, (w, h))
					targetLetter_rot = cv2.warpAffine(targetLetter_gray, M, (w, h))

					# Convert images to masks
					ret, targetShape_mask = cv2.threshold(targetShape_rot, 200, 255, cv2.THRESH_BINARY)
					ret, targetLetter_mask = cv2.threshold(targetLetter_rot, 200, 255, cv2.THRESH_BINARY)
					blank = np.full(targetBg.shape, 255, dtype=(np.uint8))


					# Construct target
					target_top = cv2.bitwise_and(blank, targetLetterColor, mask=targetLetter_mask)
					target_mid = cv2.bitwise_and(blank, targetColor, mask=cv2.bitwise_and(targetShape_mask, cv2.bitwise_not(targetLetter_mask)))
					target_bot = cv2.bitwise_and(targetBg, targetBg, mask=cv2.bitwise_not(targetShape_mask))
					target = cv2.bitwise_or(target_top, target_mid)
					target = cv2.bitwise_or(target, target_bot)
					
					if DEBUG:
						cv2.imshow("Top", target_top)
						cv2.imshow("Mid", target_mid)
						cv2.imshow("Bot", target_bot)   

					# Resize images
					if args.size is not None:
						target = cv2.resize(target, (args.size, args.size), interpolation = cv2.INTER_AREA)

					# Blur targets
					target = cv2.GaussianBlur(target, (5,5), 0)

					# Save target
					directory = '../generated/'
					if not (os.path.isdir(directory)):
						os.mkdir(directory)
					filename = background[0] + '_' + color[0] + '_' + \
								shape[0] + '_' + letterColor[0] + '_' + \
																	letter[0] + '_' + str(args.size)
					if args.format == 'json' or args.format == 'all':
						# save as JSON with labels
						packet = {}
						packet["image"] = filename + '.jpg'
						packet["shape"] = shape[0]
						packet["background_color"] = color[0]
						packet["alphanumeric"] = letter[0]
						packet["alphanumeric_color"] = letterColor[0]
						packet["orientation"] = orientation
						with open(directory + filename + '.json', 'w') as outfile:
							json.dump(packet, outfile)
					if args.format == 'jpg' or args.format == 'all':
						cv2.imwrite(directory + filename + '.jpg', target)
					else:
						print("No output format specified")
					if DEBUG:
						cv2.imshow("Debug", target)
						cv2.waitKey(0)
						cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

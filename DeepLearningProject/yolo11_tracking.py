import argparse
from ultralytics import YOLO


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('arg1')
	args = parser.parse_args()

	model = YOLO("yolo11n.pt")


	results = model.track(args.arg1, save=True, show=True)
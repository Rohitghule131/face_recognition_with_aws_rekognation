import os
from rekognition import detect_faces, compare_faces, analyze_faces, check_liveness

image_path = os.path

detect_faces(str(image_path) + "/tom_cruise.png")

compare_faces(str(image_path) + "/tom_cruise_action.png", str(image_path) + "/tom_cruise_party.png")

analyze_faces(str(image_path) + "/tom_cruise_action.png")

check_liveness(str(image_path) + "/tom_cruise_action.png")



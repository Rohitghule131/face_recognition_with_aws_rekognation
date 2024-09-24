import os
import boto3
from dotenv import load_dotenv

load_dotenv()

client_session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION_NAME")
)

rekognition = client_session.client("rekognition")


def detect_faces(image_name):
    # Load image as bytes
    with open(image_name, 'rb') as img:
        image_bytes = img.read()

    response = rekognition.detect_faces(
        Image={'Bytes': image_bytes},
        Attributes=['ALL']
    )

    face_details = response['FaceDetails']
    for face in face_details:
        print(f"Detected face with confidence {face['Confidence']}")
        print(f"Smile: {face['Smile']['Value']}")
        print(f"Emotions: {face['Emotions']}")
        print("-----")

    return face_details


def compare_faces(source_image_name, target_image_name):
    with open(source_image_name, 'rb') as source_img:
        source_image_bytes = source_img.read()

    with open(target_image_name, 'rb') as target_img:
        target_image_bytes = target_img.read()

    response = rekognition.compare_faces(
        SourceImage={'Bytes': source_image_bytes},
        TargetImage={'Bytes': target_image_bytes},
        SimilarityThreshold=70
    )

    print("RESPONSE <><><>", response)

    for face_match in response['FaceMatches']:
        similarity = face_match['Similarity']
        print(f"Face match with {similarity}% similarity")

    if not response['FaceMatches']:
        print("No faces matched")

    return response['FaceMatches']


def analyze_faces(image_name):
    with open(image_name, 'rb') as img:
        image_bytes = img.read()

    response = rekognition.detect_faces(
        Image={'Bytes': image_bytes},
        Attributes=['ALL']
    )

    face_details = response['FaceDetails']

    for face in face_details:
        print(f"Confidence: {face['Confidence']}%")
        print(f"Age Range: {face['AgeRange']['Low']} - {face['AgeRange']['High']}")
        print(f"Emotions: {[emotion['Type'] for emotion in face['Emotions'] if emotion['Confidence'] > 80]}")
        print(f"Gender: {face['Gender']['Value']}")
        print(f"Smile: {face['Smile']['Value']} ({face['Smile']['Confidence']}% confidence)")
        print(f"Eyes Open: {face['EyesOpen']['Value']} ({face['EyesOpen']['Confidence']}% confidence)")
        print(f"Mouth Open: {face['MouthOpen']['Value']} ({face['MouthOpen']['Confidence']}% confidence)")
        print("-----")

    return face_details


def check_liveness(image_name):
    with open(image_name, 'rb') as img1:
        image_bytes = img1.read()

    response = rekognition.detect_faces(
        Image={'Bytes': image_bytes},
        Attributes=['ALL']  # This provides all facial attributes
    )

    if not response['FaceDetails']:
        print("No face detected.")
        return None

    face_details = response['FaceDetails'][0]

    eyes_open = face_details['EyesOpen']['Value']
    mouth_open = face_details['MouthOpen']['Value']
    pose = face_details['Pose']

    print(f"Eyes Open: {eyes_open}")
    print(f"Mouth Open: {mouth_open}")
    print(f"Head Pose: Pitch {pose['Pitch']}, Roll {pose['Roll']}, Yaw {pose['Yaw']}")

    return {
        'eyes_open': eyes_open,
        'mouth_open': mouth_open,
        'pose': pose
    }


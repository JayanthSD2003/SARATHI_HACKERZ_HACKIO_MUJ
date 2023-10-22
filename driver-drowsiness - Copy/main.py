import time
import cv2
import dlib
import pygame
from imutils import face_utils
from scipy.spatial import distance
import json
import struct
import pvporcupine
import pyaudio
import pyttsx3
import spacy
import speech_recognition as sr


# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")


def extract_intents(user_input):
    user_input = user_input.lower()
    user_doc = nlp(user_input)
    intents = set()
    for token in user_doc:
        if not token.is_punct and not token.is_space:
            intents.add(token.text)
    return intents


def speak_response(response_text):
    engine = pyttsx3.init()
    engine.say(response_text)
    engine.runAndWait()



# Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

# Minimum threshold of eye aspect ratio below which alarm is triggered
EYE_ASPECT_RATIO_THRESHOLD = 0.2

# Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

# Counts no. of consecutive frames below threshold value
COUNTER = 0

# Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Load the JSON data
with open('cms.json', 'r') as json_file:
    commands = json.load(json_file)["commands"]

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")


# This function calculates and returns eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2 * C)
    return ear


# Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Start webcam video capture
video_capture = cv2.VideoCapture(0)

# Initialize Porcupine for wake word detection
porcupine = pvporcupine.create(access_key="PnNnuG2liwLvY7MgKXtLfK8HrTkC0vHC/7vz+q5Obxe7dkyuJi2A+w==",
                               keywords=["bumblebee"],
                               keyword_paths=['BumbleBee_en_windows_v2_2_0.ppn'])

pa = pyaudio.PyAudio()

audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length)

recognizer = sr.Recognizer()

# Give some time for camera to initialize (not required)
time.sleep(2)

while True:
    # Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect facial points through detector function
    faces = detector(gray, 0)

    # Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around each face detected
    for (x, y, w, h) in face_rectangle:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect facial points
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        # Use hull to remove convex contour discrepancies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Detect if eye aspect ratio is less than the threshold
        if eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD:
            COUNTER += 1
            # If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "You are Drowsy", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                time.sleep(5)
                COUNTER = 0

        else:
            pygame.mixer.music.stop()
            COUNTER = 0

    # Show video feed
    cv2.imshow('Video', frame)

    # Check for the wake word and respond
    pcm = audio_stream.read(porcupine.frame_length)
    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

    keyword_index = porcupine.process(pcm)

    if keyword_index >= 0:
        print("Hotword Detected")
        print("Listening for user input...")
        audio_stream.stop_stream()

        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source)
                user_input = recognizer.recognize_google(audio)

                intents = extract_intents(user_input)

                matching_response = None
                for command in commands:
                    question = command["question"].lower()
                    for intent in intents:
                        if intent in question:
                            matching_response = command["response"]
                            break
                    if matching_response:
                        break

                if matching_response:
                    print("User said:", user_input)
                    print("Assistant Response:", matching_response)
                    speak_response(matching_response)  # Speak the response

                else:
                    print("User said:", user_input)
                    print("No matching response found.")
                    speak_response("No matching response found.")  # Speak the response

            except sr.UnknownValueError:
                print("Sorry, I didn't catch that.")
                speak_response("Sorry, I didn't catch that.")  # Speak the response

            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                speak_response(f"Could not request results; {e}")  # Speak the response

        audio_stream.start_stream()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()

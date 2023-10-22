import json
import struct

import pvporcupine
import pyaudio
import pyttsx3
import spacy
import speech_recognition as sr

with open('cms.json', 'r') as json_file:
    commands = json.load(json_file)["commands"]

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


def main():
    try:
        porcupine = pvporcupine.create(access_key="PnNnuG2liwLvY7MgKXtLfK8HrTkC0vHC/7vz+q5Obxe7dkyuJi2A+w==",
                                       keywords=["bumblebee", "hey sara"],
                                       keyword_paths=['BumbleBee_en_windows_v2_2_0.ppn',
                                                      'hey-SARA_en_windows_v2_2_0.ppn'])

        pa = pyaudio.PyAudio()

        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length)

        recognizer = sr.Recognizer()

        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print("Hotword Detected")
                speak_response("Thats me how can i help you !")  # Speak the response
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
                            speak_response("No matching response found.")
                    except sr.UnknownValueError:
                        print("Sorry, I didn't catch that.")
                        speak_response("Sorry, I didn't catch that.")

                    except sr.RequestError as e:
                        print(f"Could not request results; {e}")
                        speak_response(f"Could not request results; {e}")
                audio_stream.start_stream()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()

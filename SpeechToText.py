import threading
import speech_recognition as sr
import cv2
import time

cap = cv2.VideoCapture(0)
r = sr.Recognizer()
mic = sr.Microphone()

# time of program execution.
program_time = 10


def audio_read():

    with mic as source:
        start = time.time()
        r.adjust_for_ambient_noise(source)
        print("Started Recording")
        audio = r.listen(source, phrase_time_limit=program_time, timeout=program_time)
        end = time.time()
        print(f"Recorded for {end - start} secs")

        print("\nProgram terminated")
        print("\n GENERATING LOGS ... \n")

        # recognizing the audio and writing into a file
        try:
            text_log = r.recognize_google(audio)
        except sr.UnknownValueError:
            text_log = ""

        except sr.RequestError as e:
            text_log = "{0}".format(e)
        except KeyboardInterrupt:
            pass
        
        # writing english logs
        with open("sound_log_english.txt","w+") as f:
                f.write("SOUND LOGS\n\n")
                f.write(text_log)
                f.write(" ")
                f.close()
        
        return text_log
        
# if __name__ == "__main__":
#     text_log = audio_read()
#     print(text_log)












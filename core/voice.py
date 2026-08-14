import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 180)

def speak(text):
    print(f"SALLY: {text}")
    engine.say(text)
    engine.runAndWait()

import speech_recognition as sr
from gtts import gTTS
from translation import evaluate
from SentimentAnalysis import predict
import os
from time import sleep
import pyglet
from sarcasm import predict_text
import playsound
def translate():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    # import pdb; pdb.set_trace()
    text = r.recognize_google(audio)
    sentiment = predict(text)
    translation = evaluate(text)[0][:-6]
    sarcasm = predict_text(text)
    tts = gTTS(text=translation, lang='fr')
    filename = 'temp.mp3'
    tts.save(filename)
    playsound.playsound(filename, True)
    os.remove(filename) #remove temperory file
    print(f"English: {text}, French: {translation}, Sentiment:{sentiment}, Sarcasm:{sarcasm}")
    return {"English": text, "French": translation, "Sentiment": sentiment, "Sarcasm": sarcasm}

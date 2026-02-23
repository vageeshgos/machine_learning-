import speech_recognition as sr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import webbrowser
import pyttsx3
import requests
import json
import time
from datetime import datetime, timedelta

from yt_dlp import YoutubeDL

# 🔹 Initialize recognizer & TTS engine
recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.5  # Lowered threshold for better responsiveness
engine = pyttsx3.init()

# 🔹 API details
API_KEY = "a0c879e7bde0f690d55928179cbdcd6e"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# 🔹 Paths to your schedule JSON files
schedule_path = r"C:\Users\vagee\OneDrive\Desktop\30 master ai\poject\jarvis_env\gyani_schedule.json"
roadmap_path = r"C:\Users\vagee\Downloads\AI_Agent_Placement_Roadmap.json"

# 🔹 Month dictionary for parsing dates
MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
    "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12"
}

def speak(text):
    """Convert text to speech"""
    print(f"Gyani: {text}")
    engine.say(text)
    engine.runAndWait()

def record_audio(duration=5, samplerate=16000):
    """Record audio when needed"""
    print("Listening... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    wav.write("temp_audio.wav", samplerate, audio_data)
    return "temp_audio.wav"

def recognize_audio():
    """Use SpeechRecognition to transcribe audio"""
    retries = 3  # Retry limit
    for _ in range(retries):
        audio_file = record_audio()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"User said: {command}")
            return command
        except sr.UnknownValueError:
            print("Didn't understand, trying again...")
        except sr.RequestError:
            return "Error connecting to speech service."
    return ""  # Return empty after retries

def listen_for_wake_word():
    """Continuously listen for wake word 'Gyani'"""
    while True:
        print("Listening for wake word...")
        command = recognize_audio()
        print(f"DEBUG: Heard '{command}'")  
        if "gyani" in command:
            speak("Yes, how can I help you?")
            user_command = recognize_audio()
            print(f"Received Command: {user_command}")
            process_command(user_command)

def load_schedule(path):
    """Load JSON schedule"""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"error": "Schedule file not found!"}

def get_schedule_for_today():
    """Fetch today's schedule from the roadmap"""
    roadmap = load_schedule(roadmap_path)
    
    if "error" in roadmap:
        return roadmap["error"]

    today_date = datetime.today().strftime("%B %d")  # Example: "May 5"
    
    phase_2 = roadmap.get("Phase 2 (May - June 2025)", {})
    daily_schedule = phase_2.get("Daily Schedule", {})

    if daily_schedule:
        return f"📅 {today_date}\n🎓 Exam Prep: {daily_schedule.get('Afternoon (3:00 PM - 5:00 PM)', 'Not Scheduled')}\n🤖 AI Learning: {daily_schedule.get('Evening (7:30 PM - 11:30 PM)', 'Not Scheduled')}"
    
    return f"📅 No specific schedule found for {today_date}."

def get_this_week_schedule():
    """Fetch the weekly schedule"""
    roadmap = load_schedule(roadmap_path)

    if "error" in roadmap:
        return roadmap["error"]

    today = datetime.today()
    start_of_week = today - timedelta(days=today.weekday())  
    end_of_week = start_of_week + timedelta(days=6)  

    phase_2 = roadmap.get("Phase 2 (May - June 2025)", {})
    daily_schedule = phase_2.get("Daily Schedule", {})

    if not daily_schedule:
        return "No weekly schedule available."

    schedule_text = "📅 This Week's Time Table:\n"
    
    for i in range(7):
        day = start_of_week + timedelta(days=i)
        date_str = day.strftime("%B %d")  
        schedule_text += f"\n📆 {date_str}\n🎓 Exam Prep: {daily_schedule.get('Afternoon (3:00 PM - 5:00 PM)', 'Not Scheduled')}\n🤖 AI Learning: {daily_schedule.get('Evening (7:30 PM - 11:30 PM)', 'Not Scheduled')}\n"

    return schedule_text

def process_command(command):
    """Process recognized commands"""
    print(f"Command: {command}")
    
    if "today's schedule" in command:
        today_schedule = get_schedule_for_today()
        speak(today_schedule)
    
    elif "this week's time table" in command or "this week schedule" in command:
        weekly_schedule = get_this_week_schedule()
        speak(weekly_schedule)
    
    elif "open youtube" in command:
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")
    
    elif "open google" in command:
        speak("Opening Google")
        webbrowser.open("https://www.google.com")
    
    elif "weather" in command:
        speak("Which city's weather would you like to check?")
        city = recognize_audio()
        weather_info = get_weather(city)
        speak(weather_info)
    
    elif "play song" in command:
        speak("Which song would you like to play?")
        song_name = recognize_audio()
        play_music(song_name)
    
    elif "exit" in command:
        speak("Goodbye!")
        exit()
    
    else:
        speak("I didn't understand that command.")


def recognize_audio():
    """Recognize audio from the microphone"""
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"Recognized: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

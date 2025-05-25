import speech_recognition as sr
import pyautogui
import os
import webbrowser
import time

recognizer = sr.Recognizer()

# Speak-and-listen function
def listen_command():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand")
        return ""
    except sr.RequestError:
        print("Could not request results")
        return ""

# Mouse control functions
def handle_mouse_command(command):
    if "left click" in command:
        pyautogui.click()
    elif "right click" in command:
        pyautogui.click(button='right')
    elif "double click" in command:
        pyautogui.doubleClick()
    elif "scroll up" in command:
        pyautogui.scroll(500)
    elif "scroll down" in command:
        pyautogui.scroll(-500)
    elif "move up" in command:
        pyautogui.moveRel(0, -150)
    elif "move down" in command:
        pyautogui.moveRel(0, 150)
    elif "move left" in command:
        pyautogui.moveRel(-150, 0)
    elif "move right" in command:
        pyautogui.moveRel(150, 0)
    elif "screenshot" in command:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        pyautogui.screenshot(f"screenshot_{timestamp}.png")
        print("Screenshot taken")

# App/file control functions
def handle_app_command(command):
    if "open notepad" in command:
        os.system("notepad")
    elif "open chrome" in command:
        webbrowser.open("https://www.google.com")
    elif "close window" in command:
        pyautogui.hotkey('alt', 'f4')
    # elif "open file" in command:
    #     os.startfile(r"C:\\Users\\Rubi\\Desktop")  # Change path if needed

# Main voice control loop
def virtual_mouse_voice():
    print("Voice-Controlled Virtual Mouse Started")
    while True:
        command = listen_command()
        if command:
            if "exit" in command or "quit" in command:
                print("Exiting Virtual Mouse")
                break
            handle_mouse_command(command)
            handle_app_command(command)

# Run the system
virtual_mouse_voice()

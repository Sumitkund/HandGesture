
import speech_recognition as sr
import pyautogui
import os
import webbrowser
import time
import tkinter as tk
from tkinter import ttk
import threading

pyautogui.FAILSAFE=False
recognizer = sr.Recognizer()

# --- UI Setup ---
class VoiceMouseUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Voice-Controlled Virtual Mouse")
        self.window.geometry("500x300")
        self.window.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1e1e1e", foreground="#00FFAA", font=("Segoe UI", 12))

        self.input_label = ttk.Label(self.window, text="You said:", font=("Segoe UI", 14, "bold"))
        self.input_label.pack(pady=(20, 5))

        self.input_text = ttk.Label(self.window, text="Waiting for command...", wraplength=400)
        self.input_text.pack(pady=(0, 15))

        self.output_label = ttk.Label(self.window, text="Action:", font=("Segoe UI", 14, "bold"))
        self.output_label.pack(pady=(10, 5))

        self.output_text = ttk.Label(self.window, text="None yet", wraplength=400)
        self.output_text.pack()

        threading.Thread(target=self.virtual_mouse_voice, daemon=True).start()
        self.window.mainloop()

    def update_ui(self, heard, action):
        self.input_text.config(text=heard)
        self.output_text.config(text=action)

    # --- Voice Listening ---
    def listen_command(self):
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio).lower()
            return command
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""

    # --- Mouse Control ---
    def handle_mouse_command(self, command):
        if "left click" in command:
            pyautogui.click()
            return "Left click"
        elif "right click" in command:
            pyautogui.click(button='right')
            return "Right click"
        elif "double click" in command:
            pyautogui.doubleClick()
            return "Double click"
        elif "scroll up" in command:
            pyautogui.scroll(500)
            return "Scrolled up"
        elif "scroll down" in command:
            pyautogui.scroll(-500)
            return "Scrolled down"
        elif "move up" in command:
            pyautogui.moveRel(0, -100)
            return "Moved up"
        elif "move down" in command:
            pyautogui.moveRel(0, 100)
            return "Moved down"
        elif "move left" in command:
            pyautogui.moveRel(-100, 0)
            return "Moved left"
        elif "move right" in command:
            pyautogui.moveRel(100, 0)
            return "Moved right"
        elif "screenshot" in command:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            pyautogui.screenshot(f"screenshot_{timestamp}.png")
            return "Screenshot taken"
        elif any(corner in command for corner in ["top left", "top right", "bottom left", "bottom right"]):
            screen_w, screen_h = pyautogui.size()
            positions = {
                "top left": (0, 0),
                "top right": (screen_w, 0),
                "bottom left": (0, screen_h),
                "bottom right": (screen_w, screen_h)
            }
            for key, pos in positions.items():
                if key in command:
                    pyautogui.moveTo(*pos)
                    return f"Moved to {key}"
        return "Command not recognized"

    # --- App and Window Control ---
    def handle_app_command(self, command):
        if "open notepad" in command:
            os.system("notepad")
            return "Opened Notepad"
        elif "open chrome" in command:
            webbrowser.open("https://www.google.com")
            return "Opened Chrome"
        elif "close window" in command:
            pyautogui.hotkey('alt', 'f4')
            return "Closed current window"
        elif "zoom in" in command:
            pyautogui.hotkey('ctrl', '+')
            return "Zoom in window"
        elif "zoom out" in command:
            pyautogui.hotkey('ctrl', '-')
            return "Zoom out window"
        elif "reset zoom" in command:
            pyautogui.hotkey('ctrl', '0')
            return "Rest Zoom"
        elif "maximize window" in command:
            pyautogui.hotkey('win', 'up')
            return "Maximized window"
        elif "minimize window" in command:
            pyautogui.hotkey('win', 'down')
            return "Minimized window"
        elif "open calendar" in command:
            pyautogui.hotkey('win', 'alt', 'd')
            return "Opened Calendar"
        elif "open file" in command:
            os.startfile(r"C:\\Users\\Rubi\\Desktop")  # change path if needed
            return "Opened file explorer"
        return "No matching app command"

    # --- Main Loop ---
    def virtual_mouse_voice(self):
        while True:
            command = self.listen_command()
            if command:
                print(f"You said: {command}")
                if "exit" in command or "quit" in command:
                    print("Exiting Virtual Mouse")
                    self.window.destroy()
                    break
                result = self.handle_mouse_command(command)
                if result == "Command not recognized":
                    result = self.handle_app_command(command)
                self.update_ui(command, result)

VoiceMouseUI()





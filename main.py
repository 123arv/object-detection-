import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import pyttsx3
import webbrowser
import matplotlib.pyplot as plt
import datetime
import pandas as pd


engine = pyttsx3.init()
engine.setProperty('rate', 150)


thres = 0.45
nms_threshold = 0.2


classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn.readNet(weightsPath, configPath)

stop_detection = False
last_detected_class = ""
detection_lock = threading.Lock()
detection_stats = {}
current_theme = "light"
current_font_size = 12
log_file = "detection_log.txt"

def speak(text):
    def tts_worker():
        engine.say(text)
        engine.runAndWait()
    tts_thread = threading.Thread(target=tts_worker)
    tts_thread.daemon = True
    tts_thread.start()

def log_detection(detected_class):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"{timestamp} - Detected: {detected_class}\n")

def open_wikipedia(detected_class):
    query = detected_class.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{query}"
    webbrowser.open(url)

def open_google_maps(detected_class):
    query = f"shop selling {detected_class}"
    url = f"https://www.google.com/maps/search/{query}"
    webbrowser.open(url)

def start_object_detection():
    global stop_detection, last_detected_class
    stop_detection = False
    cap = cv2.VideoCapture(0)

    while not stop_detection:
        success, img = cap.read()
        if not success:
            messagebox.showerror("Error", "Failed to access camera")
            break

        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0 / 127.5, size=(320, 320), mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
        net.setInput(blob)
        output = net.forward()

        classIds = output[0, 0, :, 1].astype(int)
        confs = output[0, 0, :, 2]
        bbox = output[0, 0, :, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        bbox = bbox.astype(int)
        confs = list(confs)
        classIds = list(classIds)

        indices = cv2.dnn.NMSBoxes(bbox.tolist(), confs, thres, nms_threshold)

        if indices is not None and len(indices) > 0:
            indices = indices.flatten()

            for i in indices:
                try:
                    class_id = int(classIds[i])
                    if class_id - 1 >= 0 and class_id - 1 < len(classNames):
                        detected_class = classNames[class_id - 1].upper()
                        with detection_lock:
                            detection_stats[detected_class] = detection_stats.get(detected_class, 0) + 1
                            log_detection(detected_class)
                            if detected_class != last_detected_class:
                                last_detected_class = detected_class
                                speak(f"Detected {detected_class}")
                                update_output(f"Detected: {detected_class}")

                        box = bbox[i]
                        x, y, w, h = box[0], box[1], box[2], box[3]
                        cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
                        cv2.putText(img, detected_class, (x+10, y+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                except IndexError as e:
                    print(f'IndexError: {e}, i: {i}, classIds: {classIds}')

        cv2.imshow('Output', img)

        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'): 
            open_wikipedia(last_detected_class)
        if key == ord('n'):  
            open_google_maps(last_detected_class)
        if key == ord('q'):  
            stop_detection = True
            break

    cap.release()
    cv2.destroyAllWindows()

def update_output(text):
    output_area.insert(tk.END, text + "\n")
    output_area.see(tk.END)

def visualize_statistics():
    with detection_lock:
        objects = list(detection_stats.keys())
        counts = list(detection_stats.values())

    plt.figure(figsize=(10, 5))
    plt.bar(objects, counts, color='blue')
    plt.xlabel('Detected Objects')
    plt.ylabel('Number of Detections')
    plt.title('Detection Statistics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def export_statistics():
    with detection_lock:
        df = pd.DataFrame(detection_stats.items(), columns=['Object', 'Count'])
        df.to_csv('detection_statistics.csv', index=False)
        messagebox.showinfo("Export Successful", "Statistics exported to detection_statistics.csv")

def start_detection_thread():
    detection_thread = threading.Thread(target=start_object_detection)
    detection_thread.daemon = True
    detection_thread.start()

def stop_object_detection():
    global stop_detection
    stop_detection = True

def toggle_theme():
    global current_theme
    if current_theme == "light":
        root.config(bg="#2E2E2E")
        for widget in root.winfo_children():
            if isinstance(widget, tk.Button) or isinstance(widget, tk.Label):
                widget.config(bg="#3E3E3E", fg="white")
            elif isinstance(widget, scrolledtext.ScrolledText):
                widget.config(bg="darkgray", fg="white")
        current_theme = "dark"
    else:
        root.config(bg="white")
        for widget in root.winfo_children():
            if isinstance(widget, tk.Button) or isinstance(widget, tk.Label):
                widget.config(bg="white", fg="black")
            elif isinstance(widget, scrolledtext.ScrolledText):
                widget.config(bg="white", fg="black")
        current_theme = "light"

def open_settings():
    settings_window = tk.Toplevel(root)
    settings_window.title("Settings")
    settings_window.geometry("300x300")

   
    color_label = tk.Label(settings_window, text="Choose Background Color:")
    color_label.pack(pady=10)

    color_var = tk.StringVar(value=current_theme)
    light_color = tk.Radiobutton(settings_window, text="Light", variable=color_var, value="light", command=lambda: update_theme("light"))
    dark_color = tk.Radiobutton(settings_window, text="Dark", variable=color_var, value="dark", command=lambda: update_theme("dark"))
    light_color.pack()
    dark_color.pack()

  
    font_label = tk.Label(settings_window, text="Choose Font Size:")
    font_label.pack(pady=10)

    font_size_var = tk.IntVar(value=current_font_size)
    for size in [10, 12, 14, 16, 18]:
        font_radio = tk.Radiobutton(settings_window, text=str(size), variable=font_size_var, value=size, command=lambda size=size: update_font(size))
        font_radio.pack()

def update_theme(theme):
    global current_theme
    if theme == "light":
        root.config(bg="white")
        for widget in root.winfo_children():
            if isinstance(widget, tk.Button) or isinstance(widget, tk.Label):
                widget.config(bg="white", fg="black")
            elif isinstance(widget, scrolledtext.ScrolledText):
                widget.config(bg="white", fg="black")
        current_theme = "light"
    else:
        root.config(bg="#2E2E2E")
        for widget in root.winfo_children():
            if isinstance(widget, tk.Button) or isinstance(widget, tk.Label):
                widget.config(bg="#3E3E3E", fg="white")
            elif isinstance(widget, scrolledtext.ScrolledText):
                widget.config(bg="darkgray", fg="white")
        current_theme = "dark"

def update_font(size):
    global current_font_size
    current_font_size = size
    output_area.config(font=("Helvetica", size))
    welcome_label.config(font=("Helvetica", size))
    start_button.config(font=("Helvetica", size))
    stop_button.config(font=("Helvetica", size))
    visualize_button.config(font=("Helvetica", size))
    export_button.config(font=("Helvetica", size))
    theme_button.config(font=("Helvetica", size))
    help_button.config(font=("Helvetica", size))

def show_help():
    help_text = (
        "Object Detection Application Help:\n\n"
        "1. Start Detection: Click 'Start Detection' to begin object detection with your webcam.\n"
        "2. Stop Detection: Click 'Stop Detection' to stop object detection.\n"
        "3. Visualize Statistics: Click 'Visualize Statistics' to see a bar chart of detected objects.\n"
        "4. Export Statistics: Click 'Export Statistics' to save detection data to a CSV file.\n"
        "5. Toggle Theme: Click 'Toggle Theme' to switch between light and dark mode.\n"
        "6. Settings: Click 'Settings' to adjust font size and background color.\n"
        "7. Help: Click 'Help' to view this guide again.\n"
        "8. Press 'C' to open Wikipedia page for the last detected object.\n"
        "9. Press 'N' to open Google Maps search for the last detected object.\n"
        "10. Press 'Q' to stop object detection.\n"
    )
    messagebox.showinfo("Help", help_text)


root = tk.Tk()
root.title("Object Detection Application")
root.geometry("800x600")


welcome_label = tk.Label(root, text="Welcome to Object Detection App", font=("Helvetica", 18))
welcome_label.pack(pady=10)


output_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Helvetica", current_font_size))
output_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)


start_button = tk.Button(root, text="Start Detection", command=start_detection_thread)
start_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(root, text="Stop Detection", command=stop_object_detection)
stop_button.pack(side=tk.LEFT, padx=10)

visualize_button = tk.Button(root, text="Visualize Statistics", command=visualize_statistics)
visualize_button.pack(side=tk.LEFT, padx=10)

export_button = tk.Button(root, text="Export Statistics", command=export_statistics)
export_button.pack(side=tk.LEFT, padx=10)

theme_button = tk.Button(root, text="Toggle Theme", command=toggle_theme)
theme_button.pack(side=tk.LEFT, padx=10)

settings_button = tk.Button(root, text="Settings", command=open_settings)
settings_button.pack(side=tk.LEFT, padx=10)

help_button = tk.Button(root, text="Help", command=show_help)
help_button.pack(side=tk.LEFT, padx=10)


root.mainloop()

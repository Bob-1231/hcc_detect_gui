import cv2
import numpy as np
import os
import tkinter as tk
import re
import threading
from PIL import Image, ImageTk
from tkinter import filedialog, Label, ttk, scrolledtext
from ultralytics import YOLO
from transformers import BartTokenizer, BartForConditionalGeneration


def find_file(root_path, target_file):
    for root, dirs, files in os.walk(root_path):
        if target_file in files:
            return os.path.join(root, target_file)
    return None


class YOLOv8GUI:
    def __init__(self, root):
        self.root = root
        self.root.state("zoomed")
        self.root.title("HCC Detection")

        self.model_liver = YOLO('./weights/liver.pt')
        self.model_hcc = YOLO('./weights/hcc.pt')

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

        self.setup_ui()


    def setup_ui(self):
        self.select_button = tk.Button(self.root, text="Select CT Image", command=self.select_file_and_summarize)
        self.select_button.pack(pady=10)

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.main_frame, width=750, height=750)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(self.main_frame, width=750, height=750)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.original_image_container = tk.Label(self.left_frame)
        self.original_image_container.pack(fill=tk.BOTH, expand=True)

        self.prediction_image_container = tk.Label(self.right_frame)
        self.prediction_image_container.pack(fill=tk.BOTH, expand=True)

        self.hcc_info_label = Label(self.info_frame, text="", font=("Times New Roman", 14))
        self.hcc_info_label.pack()

        self.summary_text = scrolledtext.ScrolledText(self.root, width=300, height=10)
        self.summary_text.pack(pady=10)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=200, mode="indeterminate")
        self.progress.pack(pady=20)
    

    def select_file_and_summarize(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            # Start image detection
            self.summary_text.delete(1.0, tk.END)
            self.progress.start(10)
            threading.Thread(target=self.detect_liver_and_hcc, args=(file_path,)).start()

            image_name = os.path.splitext(os.path.basename(file_path))[0]
            report_file_name = f"{image_name}.txt"
            root_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory where the script (this gui.py file) is located
            report_path = find_file(root_path, report_file_name)
            if report_path:
                threading.Thread(target=self.summarize_text, args=(report_path,)).start()
            else:
                self.summary_text.insert(tk.END, "No corresponding report found.")
                self.progress.stop()

            
            # Automatically find and summarize the corresponding report
            base_name = os.path.basename(file_path)
            report_file_name = os.path.splitext(base_name)[0] + '.txt'
            report_path = os.path.join("./reports", report_file_name)



    def find_sentences_with_keywords(self, text, keywords):
        pattern = re.compile(r'([^.]*\b(?:' + '|'.join(keywords) + r')\b[^.]*\.)', re.IGNORECASE)
        matches = pattern.findall(text)
        return " ".join(matches)
    
    
    def summarize_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        keywords = ["pTNM", "pT", "pNx"]
        important_sentences = self.find_sentences_with_keywords(text, keywords)
        summary = self.generate_summary(text)
        combined_summary = summary + " " + important_sentences

        self.summary_text.insert(tk.END, combined_summary)
        self.progress.stop()


    def generate_summary(self, text):
        inputs = self.tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


    def detect_liver_and_hcc(self, file_path):
        original_img = Image.open(file_path)
        original_img.thumbnail((750, 750))
        original_img_tk = ImageTk.PhotoImage(original_img)
        self.original_image_container.configure(image=original_img_tk)
        self.original_image_container.image = original_img_tk

        img_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)

        # Step 1: Liver detection
        results_liver = self.model_liver.predict(img_cv, classes=0)
        liver_detected = len(results_liver[0].boxes.xyxy.tolist()) > 0

        dimmed_img_cv = img_cv.copy()
        # Dim the whole image
        dimmed_img_cv = (dimmed_img_cv * 0.3).astype(np.uint8)

        if liver_detected:
            liver_box = results_liver[0].boxes.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, liver_box)
            cv2.rectangle(dimmed_img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for Liver
            # Copy liver area from original image to dimmed image
            dimmed_img_cv[y1:y2, x1:x2] = img_cv[y1:y2, x1:x2]

            # Step 2: HCC detection
            results_hcc = self.model_hcc.predict(img_cv, classes=1)
            hcc_detected = len(results_hcc[0].boxes.xyxy.tolist()) > 0

            if hcc_detected:
                hcc_box = results_hcc[0].boxes.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, hcc_box)
                cv2.rectangle(dimmed_img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for HCC
                confidence_score = results_hcc[0].boxes.conf.tolist()[0]
                self.hcc_info_label.config(text=f"HCC Possibility: {confidence_score:.2f}")
            else:
                self.hcc_info_label.config(text="No HCC ROI Detected")

            # Convert back to PIL Image to display in tkinter
            final_img = Image.fromarray(cv2.cvtColor(dimmed_img_cv, cv2.COLOR_BGR2RGB))
            final_img_tk = ImageTk.PhotoImage(final_img)
            self.prediction_image_container.configure(image=final_img_tk)
            self.prediction_image_container.image = final_img_tk

        else:
            results_hcc = self.model_hcc.predict(img_cv, classes=1)
            hcc_detected = len(results_hcc[0].boxes.xyxy.tolist()) > 0

            if hcc_detected:
                hcc_box = results_hcc[0].boxes.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, hcc_box)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for HCC
                confidence_score = results_hcc[0].boxes.conf.tolist()[0]
                self.hcc_info_label.config(text=f"HCC Possibility: {confidence_score:.2f}")
            else:
                self.hcc_info_label.config(text="No HCC ROI Detected")

            # Convert back to PIL Image to display in tkinter
            final_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            final_img_tk = ImageTk.PhotoImage(final_img)
            self.prediction_image_container.configure(image=final_img_tk)
            self.prediction_image_container.image = final_img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv8GUI(root)
    root.mainloop()

# ==============================================================================
# SIGNSENSE AI - Sign Language to Text & Speech Conversion
# Enhanced Modern UI & Precision Inference Engine
# ==============================================================================
import os
import sys
import time
import operator
import threading
from string import ascii_uppercase

import cv2
import numpy as np
from PIL import Image, ImageTk
import enchant

import tkinter as tk
from tkinter import ttk

# Optional ttkbootstrap for a modern flat theme; fallback gracefully to standard ttk
try:
    import ttkbootstrap as tb
    HAS_TB = True
except ImportError:
    HAS_TB = False

# Optional pyttsx3 for Voice / Speech output
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

from tensorflow.keras.models import model_from_json, Sequential

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


class Application:

    def __init__(self):
        # 1. Initialize Dictionary & Text-to-Speech
        try:
            self.hs = enchant.Dict('en_US')
        except Exception:
            self.hs = None

        self.tts_lock = threading.Lock()

        # 2. Robust Model Path Resolution
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "Models")
        if not os.path.exists(models_dir):
            models_dir = os.path.join(base_dir, "Sign-Language-To-Text-Conversion", "Models")

        # Load Primary Model
        with open(os.path.join(models_dir, "model_new.json"), "r") as f:
            self.model_json = f.read()
        self.loaded_model = model_from_json(self.model_json, custom_objects={'Sequential': Sequential})
        self.loaded_model.load_weights(os.path.join(models_dir, "model_new.h5"))

        # Load Layer 2 Submodel: DRU
        with open(os.path.join(models_dir, "model-bw_dru.json"), "r") as f:
            self.model_json_dru = f.read()
        self.loaded_model_dru = model_from_json(self.model_json_dru, custom_objects={'Sequential': Sequential})
        self.loaded_model_dru.load_weights(os.path.join(models_dir, "model-bw_dru.h5"))

        # Load Layer 2 Submodel: TKDI
        with open(os.path.join(models_dir, "model-bw_tkdi.json"), "r") as f:
            self.model_json_tkdi = f.read()
        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi, custom_objects={'Sequential': Sequential})
        self.loaded_model_tkdi.load_weights(os.path.join(models_dir, "model-bw_tkdi.h5"))

        # Load Layer 2 Submodel: SMN
        with open(os.path.join(models_dir, "model-bw_smn.json"), "r") as f:
            self.model_json_smn = f.read()
        self.loaded_model_smn = model_from_json(self.model_json_smn, custom_objects={'Sequential': Sequential})
        self.loaded_model_smn.load_weights(os.path.join(models_dir, "model-bw_smn.h5"))

        print("Loaded all models successfully from disk")

        # 3. State Variables & Recognition Tracking
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        self.threshold_min_value = 70  # Default Otsu/adaptive threshold offset
        self.frame_skip_counter = 0
        self.predict_every_n_frames = 2  # Balanced latency and high FPS

        self.ct = {'blank': 0}
        for i in ascii_uppercase:
            self.ct[i] = 0
        self.blank_flag = 0

        self.current_symbol = "Empty"
        self.current_confidence = 0.0
        self.word = ""
        self.str = ""
        self.hold_threshold = 24  # Frames required to commit a letter (~0.8s for smooth typing)

        # 4. Construct Modern GUI
        self.setup_ui()

        # Start Camera Loop
        self.video_loop()

    def setup_ui(self):
        if HAS_TB:
            self.root = tb.Window(themename="darkly")
        else:
            self.root = tk.Tk()
            self.root.configure(bg="#1e1e2f")

        self.root.title("SignSense AI - Sign Language to Text & Speech")
        self.root.geometry("1120x800")
        self.root.minsize(980, 720)
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        # Main Scrollable / Packed Container
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header Banner
        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X, pady=(0, 10))

        title_lbl = ttk.Label(
            header,
            text="SIGNSENSE AI",
            font=("Segoe UI", 22, "bold")
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = ttk.Label(
            header,
            text=" • Real-Time ASL Recognition & Assistive Translation",
            font=("Segoe UI", 12)
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=(6, 0), pady=(7, 0))

        status_badge = ttk.Label(
            header,
            text="● Live Feed",
            font=("Segoe UI", 11, "bold")
        )
        status_badge.pack(side=tk.RIGHT, pady=5)

        # Content Split: Left (Vision Feeds) & Right (Workspace & Controls)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # ----------------- LEFT COLUMN: VISION -----------------
        left_col = ttk.LabelFrame(content_frame, text=" Live Camera Feed & Hand ROI ", padding=10)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # Camera Panel
        self.panel = ttk.Label(left_col, background="#0f172a")
        self.panel.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        # Bottom row inside left column: Silhouette Mask + Lighting Slider
        vision_controls = ttk.Frame(left_col)
        vision_controls.pack(fill=tk.X, pady=(4, 0))

        # Silhouette Mask Frame
        mask_box = ttk.LabelFrame(vision_controls, text=" Processed Silhouette (AI Input) ", padding=4)
        mask_box.pack(side=tk.LEFT, padx=(0, 10))

        self.panel2 = ttk.Label(mask_box, background="#000000", width=22)
        self.panel2.pack(padx=2, pady=2)

        # Slider Box for Lighting Adjustment
        slider_box = ttk.LabelFrame(vision_controls, text=" Ambient Light / Sensitivity ", padding=8)
        slider_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.slider_lbl = ttk.Label(slider_box, text=f"Threshold Level: {self.threshold_min_value}")
        self.slider_lbl.pack(anchor=tk.W)

        self.thresh_scale = ttk.Scale(
            slider_box,
            from_=30,
            to=150,
            value=self.threshold_min_value,
            command=self.update_threshold
        )
        self.thresh_scale.pack(fill=tk.X, pady=(4, 2))

        ttk.Label(
            slider_box,
            text="Adjust slider until hand silhouette is distinct with minimal background noise.",
            font=("Segoe UI", 8),
            wraplength=220
        ).pack(anchor=tk.W)

        # ----------------- RIGHT COLUMN: RECOGNITION & WORKSPACE -----------------
        right_col = ttk.Frame(content_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        # 1. Detection Card
        detect_card = ttk.LabelFrame(right_col, text=" Real-Time Detection ", padding=12)
        detect_card.pack(fill=tk.X, pady=(0, 10))

        d_header = ttk.Frame(detect_card)
        d_header.pack(fill=tk.X)

        ttk.Label(d_header, text="Current Symbol:", font=("Segoe UI", 12)).pack(side=tk.LEFT)
        self.conf_lbl = ttk.Label(d_header, text="Confidence: 0.0%", font=("Segoe UI", 11, "bold"))
        self.conf_lbl.pack(side=tk.RIGHT)

        # Large Symbol Display
        self.panel3 = ttk.Label(
            detect_card,
            text="--",
            font=("Segoe UI", 48, "bold"),
            anchor=tk.CENTER
        )
        self.panel3.pack(fill=tk.X, pady=6)

        # Progress bar showing hold time towards committing character
        self.hold_progress = ttk.Progressbar(detect_card, maximum=100, value=0)
        self.hold_progress.pack(fill=tk.X, pady=(2, 0))

        # 2. Text Construction Card
        text_card = ttk.LabelFrame(right_col, text=" Sentence & Word Construction ", padding=12)
        text_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(text_card, text="Current Word:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.panel4 = ttk.Label(
            text_card,
            text="",
            font=("Segoe UI", 20, "bold")
        )
        self.panel4.pack(anchor=tk.W, pady=(2, 8))

        ttk.Label(text_card, text="Constructed Sentence:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.panel5 = ttk.Label(
            text_card,
            text="",
            font=("Segoe UI", 15),
            wraplength=420,
            justify=tk.LEFT
        )
        self.panel5.pack(anchor=tk.W, fill=tk.BOTH, expand=True, pady=(2, 8))

        # Autocomplete Suggestions
        sugg_frame = ttk.LabelFrame(text_card, text=" Autocomplete Suggestions ", padding=8)
        sugg_frame.pack(fill=tk.X, pady=(4, 0))

        btn_row = ttk.Frame(sugg_frame)
        btn_row.pack(fill=tk.X)

        self.bt1 = ttk.Button(btn_row, text="-", command=self.action1)
        self.bt1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.bt2 = ttk.Button(btn_row, text="-", command=self.action2)
        self.bt2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.bt3 = ttk.Button(btn_row, text="-", command=self.action3)
        self.bt3.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 3. Actions & Accessibility Controls
        actions_card = ttk.LabelFrame(right_col, text=" Accessibility & Quick Controls ", padding=10)
        actions_card.pack(fill=tk.X)

        ctrl_row1 = ttk.Frame(actions_card)
        ctrl_row1.pack(fill=tk.X, pady=(0, 5))

        self.speak_btn = ttk.Button(ctrl_row1, text="🔊 Speak Sentence", command=self.speak_sentence)
        self.speak_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.space_btn = ttk.Button(ctrl_row1, text="␣ Space", command=self.action_space)
        self.space_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ctrl_row2 = ttk.Frame(actions_card)
        ctrl_row2.pack(fill=tk.X)

        self.backspace_btn = ttk.Button(ctrl_row2, text="⌫ Backspace", command=self.action_backspace)
        self.backspace_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.clear_btn = ttk.Button(ctrl_row2, text="🗑 Clear Sentence", command=self.action_clear)
        self.clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

    def update_threshold(self, val):
        self.threshold_min_value = int(float(val))
        self.slider_lbl.config(text=f"Threshold Level: {self.threshold_min_value}")

    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            # Standardized ROI coordinates
            x1 = int(0.5 * w)
            y1 = 10
            x2 = w - 10
            y2 = int(0.5 * w)

            # Ensure valid bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 10, min(x2, w))
            y2 = max(y1 + 10, min(y2, h))

            # Crop ROI before drawing overlay
            roi = frame[y1:y2, x1:x2].copy()

            # Preprocessing Pipeline (matched to training pipeline)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, self.threshold_min_value, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Draw stylish ROI bounding box on camera frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 255), 2)
            cv2.putText(frame, "ROI - Place Hand", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 255), 2)

            # Run CNN Prediction (optimized frame rate)
            self.frame_skip_counter += 1
            if self.frame_skip_counter >= self.predict_every_n_frames:
                self.frame_skip_counter = 0
                self.predict(res)

            # Convert Main Frame for UI Display (resized cleanly to fit panel)
            display_frame = cv2.resize(frame, (520, 390))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # Convert Processed Silhouette for UI Preview
            res_preview = cv2.resize(res, (180, 180))
            self.current_image2 = Image.fromarray(res_preview)
            imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk2
            self.panel2.config(image=imgtk2)

            # Update Labels & Badges
            sym_text = self.current_symbol if self.current_symbol != 'blank' else "Blank"
            self.panel3.config(text=sym_text)
            self.conf_lbl.config(text=f"Confidence: {self.current_confidence * 100:.1f}%")

            # Update Hold Progress
            current_count = self.ct.get(self.current_symbol, 0)
            progress_pct = min(100.0, (current_count / float(self.hold_threshold)) * 100.0)
            self.hold_progress['value'] = progress_pct

            self.panel4.config(text=self.word if self.word else "-")
            self.panel5.config(text=self.str if self.str else "(Sentence will appear here...)")

            # Word Suggestions via PyEnchant
            if self.hs and len(self.word.strip()) > 0:
                predicts = self.hs.suggest(self.word)
            else:
                predicts = []

            self.bt1.config(text=predicts[0] if len(predicts) > 0 else "-", state=tk.NORMAL if len(predicts) > 0 else tk.DISABLED)
            self.bt2.config(text=predicts[1] if len(predicts) > 1 else "-", state=tk.NORMAL if len(predicts) > 1 else tk.DISABLED)
            self.bt3.config(text=predicts[2] if len(predicts) > 2 else "-", state=tk.NORMAL if len(predicts) > 2 else tk.DISABLED)

        self.root.after(15, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))

        # Accuracy Filter 1: Check if hand is present in ROI
        # In this dataset, blank background is > 99.2% white
        white_pixels = cv2.countNonZero(test_image)
        total_pixels = 128 * 128
        white_ratio = white_pixels / float(total_pixels)

        if white_ratio > 0.992 or white_ratio < 0.05:
            self.current_symbol = 'blank'
            self.current_confidence = 0.0
            return

        # Layer 1: Global 27-Class Prediction
        input_data = test_image.reshape(1, 128, 128, 1)
        result = self.loaded_model.predict(input_data, verbose=0)

        prediction = {'blank': float(result[0][0])}
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = float(result[0][inde])
            inde += 1

        prediction_sorted = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        top_symbol = prediction_sorted[0][0]
        top_conf = prediction_sorted[0][1]

        # Accuracy Filter 2: Confidence gating (reject low-confidence ambiguous noise)
        if top_conf < 0.35:
            self.current_symbol = 'blank'
            self.current_confidence = top_conf
            return

        self.current_symbol = top_symbol
        self.current_confidence = top_conf

        # Layer 2: Selective Disambiguation (Only run required submodel)
        if self.current_symbol in ('D', 'R', 'U'):
            result_dru = self.loaded_model_dru.predict(input_data, verbose=0)
            pred_dru = {
                'D': float(result_dru[0][0]),
                'R': float(result_dru[0][1]),
                'U': float(result_dru[0][2])
            }
            pred_dru_sorted = sorted(pred_dru.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = pred_dru_sorted[0][0]
            self.current_confidence = pred_dru_sorted[0][1]

        elif self.current_symbol in ('D', 'I', 'K', 'T'):
            result_tkdi = self.loaded_model_tkdi.predict(input_data, verbose=0)
            pred_tkdi = {
                'D': float(result_tkdi[0][0]),
                'I': float(result_tkdi[0][1]),
                'K': float(result_tkdi[0][2]),
                'T': float(result_tkdi[0][3])
            }
            pred_tkdi_sorted = sorted(pred_tkdi.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = pred_tkdi_sorted[0][0]
            self.current_confidence = pred_tkdi_sorted[0][1]

        elif self.current_symbol in ('M', 'N', 'S'):
            result_smn = self.loaded_model_smn.predict(input_data, verbose=0)
            pred_smn = {
                'M': float(result_smn[0][0]),
                'N': float(result_smn[0][1]),
                'S': float(result_smn[0][2])
            }
            pred_smn_sorted = sorted(pred_smn.items(), key=operator.itemgetter(1), reverse=True)
            # CRITICAL ACCURACY FIX: Correctly accept M, N, or S from the disambiguation submodel!
            self.current_symbol = pred_smn_sorted[0][0]
            self.current_confidence = pred_smn_sorted[0][1]

        # Temporal Accumulation & Debouncing
        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] = self.ct.get(self.current_symbol, 0) + 1

        # Commit character once held stably across hold_threshold frames
        if self.ct[self.current_symbol] > self.hold_threshold:
            # Check stability dominance against other characters
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                diff = abs(self.ct[self.current_symbol] - self.ct.get(i, 0))
                if diff <= 10:
                    self.ct['blank'] = 0
                    for k in ascii_uppercase:
                        self.ct[k] = 0
                    return

            # Reset counts upon commit
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.word) > 0:
                        if len(self.str) > 0:
                            self.str += " "
                        self.str += self.word
                        self.word = ""
            else:
                self.blank_flag = 0
                self.word += self.current_symbol

    def action1(self):
        if not self.hs:
            return
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 0:
            if len(self.str) > 0:
                self.str += " "
            self.str += predicts[0]
            self.word = ""

    def action2(self):
        if not self.hs:
            return
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 1:
            if len(self.str) > 0:
                self.str += " "
            self.str += predicts[1]
            self.word = ""

    def action3(self):
        if not self.hs:
            return
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 2:
            if len(self.str) > 0:
                self.str += " "
            self.str += predicts[2]
            self.word = ""

    def action4(self):
        if not self.hs:
            return
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 3:
            if len(self.str) > 0:
                self.str += " "
            self.str += predicts[3]
            self.word = ""

    def action5(self):
        if not self.hs:
            return
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 4:
            if len(self.str) > 0:
                self.str += " "
            self.str += predicts[4]
            self.word = ""

    def action_space(self):
        """Append current word or space to sentence."""
        if len(self.word) > 0:
            if len(self.str) > 0:
                self.str += " "
            self.str += self.word
            self.word = ""
        else:
            if len(self.str) > 0 and not self.str.endswith(" "):
                self.str += " "

    def action_backspace(self):
        """Remove last character from word or sentence."""
        if len(self.word) > 0:
            self.word = self.word[:-1]
        elif len(self.str) > 0:
            self.str = self.str[:-1]

    def action_clear(self):
        """Clear sentence and word buffers."""
        self.word = ""
        self.str = ""

    def speak_sentence(self):
        """Asynchronously vocalize the sentence using Pyttsx3 Text-To-Speech."""
        if not HAS_TTS:
            return

        text = self.str.strip()
        if not text and self.word.strip():
            text = self.word.strip()

        if not text:
            return

        def _tts():
            with self.tts_lock:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print("TTS error:", e)

        thread = threading.Thread(target=_tts, daemon=True)
        thread.start()

    def destructor(self):
        print("Closing Application...")
        try:
            self.vs.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    print("Starting SignSense Application...")
    app = Application()
    app.root.mainloop()

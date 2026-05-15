import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import glob
from ultralytics import YOLO


class RoadAnomalyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MCE 415: YOLOv3 Road Anomaly Detector")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        self.model = None
        self.image_path = None
        self.cv_image = None

        self.setup_ui()

    def setup_ui(self):
        # --- TOP FRAME: Controls ---
        control_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        control_frame.pack(fill=tk.X)

        tk.Label(control_frame, text="1. Select YOLOv3 Weight (.pt):", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(
            side=tk.LEFT, padx=10)

        # Find all .pt files in the current PyCharm directory
        self.weight_files = glob.glob("*.pt")
        self.weight_combo = ttk.Combobox(control_frame, values=self.weight_files, state="readonly", width=30)
        self.weight_combo.pack(side=tk.LEFT, padx=5)
        if self.weight_files:
            self.weight_combo.current(0)

        # Buttons
        self.btn_load_img = tk.Button(control_frame, text="2. Load Road Image", command=self.load_image, bg="#2196F3",
                                      fg="white", font=("Arial", 10, "bold"))
        self.btn_load_img.pack(side=tk.LEFT, padx=15)

        self.btn_infer = tk.Button(control_frame, text="3. Run Inference", command=self.run_inference, bg="#4CAF50",
                                   fg="white", font=("Arial", 10, "bold"))
        self.btn_infer.pack(side=tk.LEFT, padx=5)

        # --- MIDDLE FRAME: Image Display ---
        self.canvas_frame = tk.Frame(self.root, bg="black")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.image_label = tk.Label(self.canvas_frame, text="Upload an image to begin...", bg="gray", fg="white",
                                    font=("Arial", 14))
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # --- BOTTOM FRAME: Status ---
        self.status_var = tk.StringVar()
        self.status_var.set("System Ready. Waiting for user input.")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                   font=("Arial", 10))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            self.status_var.set(f"Loaded: {self.image_path}")

            # Display the original image
            img = Image.open(self.image_path)
            self.display_image(img)

    def run_inference(self):
        if not self.weight_combo.get():
            messagebox.showerror("Error", "Please select a .pt weight file from the dropdown!")
            return
        if not self.image_path:
            messagebox.showerror("Error", "Please load an image first!")
            return

        try:
            self.status_var.set("Loading YOLOv3 Model & Running Inference...")
            self.root.update()

            # 1. Load the selected model
            selected_weight = self.weight_combo.get()
            self.model = YOLO(selected_weight)

            # 2. Run Inference
            results = self.model.predict(source=self.image_path, imgsz=640, conf=0.5)

            # 3. Extract annotated image (OpenCV format: BGR)
            annotated_frame = results[0].plot()

            # 4. Convert OpenCV BGR to Pillow RGB format
            color_coverted = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            final_image = Image.fromarray(color_coverted)

            # 5. Display the result
            self.display_image(final_image)
            self.status_var.set(f"Inference Complete using {selected_weight}! Found {len(results[0].boxes)} anomalies.")

        except Exception as e:
            messagebox.showerror("Inference Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error during inference.")

    def display_image(self, img):
        # Resize image to fit the Tkinter window while maintaining aspect ratio
        display_width = 800
        display_height = 550
        img.thumbnail((display_width, display_height))

        # Tkinter requires a special PhotoImage object
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo, text="")


if __name__ == "__main__":
    # Ensure your .pt files are in the SAME folder as this script before running!
    root = tk.Tk()
    app = RoadAnomalyApp(root)
    root.mainloop()
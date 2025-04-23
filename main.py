import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

# Modern color scheme
BG_COLOR = "#f0f2f5"
PRIMARY_COLOR = "#A31621"
SECONDARY_COLOR = "#ffffff"
ACCENT_COLOR = "#36b9cc"
TEXT_COLOR = "#5a5c69"

# Load the pre-trained MobileNet V2 model from TensorFlow Hub
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                       input_shape=(224, 224, 3),
                                       trainable=False)

# Define the model architecture
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(224, 224, 3)),
    keras.layers.Lambda(lambda x: feature_extractor_layer(x)),
    keras.layers.Dense(9, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load the trained weights
model.load_weights("my_model.weights.h5")

# Define disease labels with emojis for better visualization
disease_labels = {
    0: '🦠 Cellulitis',
    1: '🦠 Impetigo',
    2: '👣 Athlete Foot',
    3: '💅 Nail Fungus',
    4: '🌀 Ringworm',
    5: '🐛 Cutaneous Larva Migrans',
    6: '🌡 Chickenpox',
    7: '🔥 Shingles',
    8: '✅ Normal Skin'
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def predict_disease(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label = disease_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_label, f"{confidence:.2f}%"

def select_image():
    file_types = [("Image files", "*.jpg *.jpeg *.png")]
    file_path = filedialog.askopenfilename(filetypes=file_types)
    
    if file_path:
        try:
            image = Image.open(file_path)
            image = image.resize((300, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # Update image display
            canvas.image = photo
            canvas.create_image(150, 150, image=photo)
            
            # Clear previous results
            result_label.config(text="")
            confidence_label.config(text="")
            
            global selected_image_path
            selected_image_path = file_path
            
            # Enable predict button
            predict_button.config(state=tk.NORMAL)
            
        except Exception as e:
            result_label.config(text=f"Error: {str(e)}", fg="red")

def predict():
    if selected_image_path:
        try:
            # Show loading state
            result_label.config(text="Analyzing...", fg=PRIMARY_COLOR)
            root.update()
            
            # Get prediction
            disease, confidence = predict_disease(selected_image_path)
            
            # Display results with styling
            result_label.config(text=f"Prediction: {disease}", 
                               fg=PRIMARY_COLOR, 
                               font=('Helvetica', 12, 'bold'))
            confidence_label.config(text=f"Confidence: {confidence}", 
                                  fg=ACCENT_COLOR,
                                  font=('Helvetica', 11))
            
        except Exception as e:
            result_label.config(text=f"Prediction Error: {str(e)}", fg="red")
    else:
        result_label.config(text="Please select an image first", fg="red")

# Create the main window with modern styling
root = tk.Tk()
root.title("Skin Disease Prediction")
root.geometry("600x700")
root.configure(bg=BG_COLOR)
root.resizable(False, False)

# Make the window slightly transparent for modern look
root.attributes('-alpha', 0.98)

# Header frame
header_frame = tk.Frame(root, bg=PRIMARY_COLOR, height=80)
header_frame.pack(fill=tk.X)
header_frame.pack_propagate(False)


# Main content frame
content_frame = tk.Frame(root, bg=SECONDARY_COLOR, padx=20, pady=20)
content_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

# Instructions
instruction_label = tk.Label(content_frame, 
                           text="Upload an image of skin condition for analysis",
                           bg=SECONDARY_COLOR,
                           fg=TEXT_COLOR,
                           font=('Helvetica', 11))
instruction_label.pack(pady=(0, 20))

# Image display canvas with border
canvas = tk.Canvas(content_frame, 
                  width=300, 
                  height=300, 
                  bg="#f8f9fc",
                  highlightbackground="#dddfeb",
                  highlightthickness=2)
canvas.pack()
canvas.create_text(150, 150, 
                  text="No image selected", 
                  fill="#b7b9cc",
                  font=('Helvetica', 10))

# Button frame
button_frame = tk.Frame(content_frame, bg=SECONDARY_COLOR)
button_frame.pack(pady=20)

# Modern buttons with hover effects
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 10), padding=10)
style.map('TButton', 
          foreground=[('pressed', 'white'), ('active', 'white')],
          background=[('pressed', '!disabled', '#3a5ccc'), ('active', '#5c85ff')])

select_button = ttk.Button(button_frame, 
                          text="📁 Select Image", 
                          command=select_image,
                          style='TButton')
select_button.pack(side=tk.LEFT, padx=10)

predict_button = ttk.Button(button_frame, 
                           text="🔍 Predict", 
                           command=predict,
                           style='TButton',
                           state=tk.DISABLED)
predict_button.pack(side=tk.LEFT, padx=10)

# Result display
result_frame = tk.Frame(content_frame, bg=SECONDARY_COLOR)
result_frame.pack(pady=10)

result_label = tk.Label(result_frame, 
                       text="", 
                       bg=SECONDARY_COLOR,
                       font=('Helvetica', 12),
                       wraplength=400)
result_label.pack()

confidence_label = tk.Label(result_frame, 
                           text="", 
                           bg=SECONDARY_COLOR,
                           font=('Helvetica', 11))
confidence_label.pack()

# Initialize global variable for selected image path
selected_image_path = None

# Run the application
root.mainloop()

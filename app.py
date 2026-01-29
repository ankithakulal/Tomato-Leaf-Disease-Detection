import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, redirect, url_for
import os
import re
import sqlite3
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import base64
from gtts import gTTS


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
AUDIO_FOLDER = 'static/audio/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# Class labels
class_names = ['Blight', 'Healthy', 'Leaf spot', 'Mildew']

# Load your trained hybrid model
model = load_model('Models/ensemble_hybrid.h5', compile=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_prepare_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_gradcam_heatmap(model, img_array, conv_layer_name='block14_sepconv2_act'):
    
    model_4 = model.get_layer('model_3')  # Adjust this to point to the correct model
    last_conv_layer = model_4.get_layer(conv_layer_name)  # Choose the correct convolutional layer

    grad_model = tf.keras.models.Model(
        [model_4.input],
        [last_conv_layer.output, model_4.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_audio(text, filename, lang='en', slow=False):
    """Generate audio file using gTTS"""
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
        tts.save(audio_path)
        return AUDIO_FOLDER + filename
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def get_prediction_text(disease_name, confidence):
    """Generate text for prediction"""
    is_healthy = disease_name.lower() == 'healthy'
    text = "Disease Detection Results. "
    if is_healthy:
        text += f"Great news! Your tomato leaf is detected as {disease_name}. "
    else:
        text += f"Disease detected: {disease_name}. "
    text += f"Confidence level: {confidence} percent. "
    return text

def get_causes_text(disease_name):
    """Generate text for disease causes"""
    causes = {
        'blight': [
            "Fungal Pathogens: Caused by Alternaria solani (Early Blight) or Phytophthora infestans (Late Blight)",
            "Environmental Factors: High humidity, warm temperatures (20-30°C), and prolonged leaf wetness",
            "Poor Air Circulation: Dense plant spacing and lack of proper ventilation",
            "Contaminated Soil: Infected plant debris and soil-borne pathogens",
            "Water Management: Overhead irrigation and water splashing from infected plants"
        ],
        'leaf spot': [
            "Bacterial Infection: Caused by Xanthomonas or Pseudomonas bacteria",
            "Wet Conditions: Prolonged leaf wetness from rain, dew, or irrigation",
            "Wound Entry: Bacteria enter through natural openings or wounds on leaves",
            "Contaminated Tools: Spread through pruning tools, equipment, or hands",
            "Seed Transmission: Infected seeds can introduce the disease"
        ],
        'mildew': [
            "Powdery Mildew Fungus: Caused by Oidium neolycopersici or Leveillula taurica",
            "Moderate Temperatures: Thrives in 20-27°C with high humidity",
            "Low Light Conditions: Shaded areas and poor sunlight exposure",
            "Dense Planting: Crowded plants reduce air circulation",
            "Nutrient Imbalance: Excessive nitrogen and low potassium levels"
        ],
        'healthy': [
            "Optimal Growing Conditions: Proper temperature, humidity, and light exposure",
            "Good Air Circulation: Adequate spacing and ventilation",
            "Balanced Nutrition: Proper fertilization and nutrient management",
            "Preventive Care: Regular monitoring and early intervention",
            "Healthy Soil: Well-drained soil with good organic matter"
        ]
    }
    
    disease_key = disease_name.lower()
    # Handle 'leaf spot' -> keep as 'leaf spot'
    if disease_key not in causes:
        disease_key = disease_key.replace(' ', '_')
    if disease_key in causes:
        text = "Disease Causes and Reasons. "
        for i, cause in enumerate(causes[disease_key], 1):
            text += f"{i}. {cause}. "
        return text
    return "Disease analysis in progress. Please consult with agricultural experts for detailed information."

def get_recommendations_text(disease_name):
    """Generate text for treatment recommendations"""
    recommendations = {
        'blight': [
            "Fungicide Application: Apply copper-based fungicides or chlorothalonil every 7-10 days",
            "Remove Infected Leaves: Prune and dispose of affected leaves immediately",
            "Improve Airflow: Increase plant spacing and use proper trellising",
            "Water Management: Use drip irrigation and avoid overhead watering",
            "Crop Rotation: Rotate with non-solanaceous crops for 2-3 years",
            "Resistant Varieties: Plant disease-resistant tomato varieties next season"
        ],
        'leaf spot': [
            "Copper-Based Sprays: Apply copper fungicides or bactericides early in the morning",
            "Sanitation: Remove and destroy infected plant material",
            "Avoid Overhead Watering: Use drip irrigation to keep leaves dry",
            "Tool Sterilization: Disinfect pruning tools between plants",
            "Biological Control: Use beneficial bacteria like Bacillus subtilis",
            "Proper Spacing: Ensure adequate plant spacing for better air circulation"
        ],
        'mildew': [
            "Sulfur-Based Fungicides: Apply sulfur or potassium bicarbonate sprays",
            "Neem Oil Treatment: Use organic neem oil as a preventive measure",
            "Improve Sunlight: Prune to increase light penetration",
            "Reduce Humidity: Ensure proper ventilation and spacing",
            "Balanced Fertilization: Reduce nitrogen and increase potassium",
            "Regular Monitoring: Check plants weekly and treat early symptoms"
        ],
        'healthy': [
            "Maintain Current Practices: Continue with existing care routine",
            "Preventive Monitoring: Regular inspection for early disease detection",
            "Proper Nutrition: Maintain balanced fertilization schedule",
            "Water Management: Continue optimal irrigation practices",
            "Sanitation: Keep garden clean and remove debris",
            "Resistant Varieties: Consider planting disease-resistant varieties"
        ]
    }
    
    disease_key = disease_name.lower()
    # Handle 'leaf spot' -> keep as 'leaf spot'
    if disease_key not in recommendations:
        disease_key = disease_key.replace(' ', '_')
    if disease_key in recommendations:
        text = "Treatment Recommendations. "
        for i, rec in enumerate(recommendations[disease_key], 1):
            text += f"{i}. {rec}. "
        return text
    return "Please consult with agricultural experts for specific treatment recommendations."

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    img_array = load_and_prepare_image(file_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    # Grad-CAM
    heatmap = get_gradcam_heatmap(model, img_array)
    img_orig = cv2.imread(file_path)
    img_orig = cv2.resize(img_orig, (128, 128))

    heatmap_resized = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_colored, 0.4, 0)

    _, buffer = cv2.imencode('.jpg', superimposed_img)
    gradcam_img_b64 = base64.b64encode(buffer).decode('utf-8')
    gradcam_img_b64 = 'data:image/jpeg;base64,' + gradcam_img_b64
    # Save the image
    cv2.imwrite('static/uploads/gradcam.jpg', superimposed_img)



    image_url = UPLOAD_FOLDER + filename
    prediction = f"Prediction: {predicted_label} with confidence {confidence*100:.2f}%"
    print(prediction)

    # Generate audio files using gTTS
    confidence_str = f"{confidence*100:.2f}"
    unique_id = filename.split('.')[0]  # Use filename without extension as unique ID
    
    # Generate prediction audio
    prediction_text = get_prediction_text(predicted_label, confidence_str)
    prediction_audio = generate_audio(prediction_text, f'prediction_{unique_id}.mp3')
    
    # Generate causes audio
    causes_text = get_causes_text(predicted_label)
    causes_audio = generate_audio(causes_text, f'causes_{unique_id}.mp3')
    
    # Generate recommendations audio
    recommendations_text = get_recommendations_text(predicted_label)
    recommendations_audio = generate_audio(recommendations_text, f'recommendations_{unique_id}.mp3')

    return render_template('result.html',
                        image_url=image_url, 
                        prediction=prediction,
                        disease_name=predicted_label,
                        gradcam_img=gradcam_img_b64,
                        confidence=confidence_str,
                        prediction_audio=prediction_audio,
                        causes_audio=causes_audio,
                        recommendations_audio=recommendations_audio
                        )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/graphs')
def graphs():
	return render_template('graphs.html')


  
if __name__ == '__main__':
    app.run(debug=False)

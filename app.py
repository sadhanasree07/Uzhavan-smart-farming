from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import joblib
import pandas as pd
import numpy as np
import bcrypt
import time
import google.generativeai as genai
import logging
import random
from gtts import gTTS
import sklearn
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
from datetime import datetime
import importlib.util
import re

# Load environment variables
load_dotenv()

# Import all_states from data.schemes
try:
    from data.schemes import all_states
except ImportError:
    logging.error("data.schemes module not found. Using placeholder data.")
    all_states = [{
        "state": "tamil nadu",
        "eligibility": "Farmers with land ownership",
        "available_schemes": ["PM-Kisan", "Crop Insurance"],
        "contact": "Local agriculture office"
    }]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize Flask app
app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24).hex())
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- MongoDB Configuration ---
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
   logging.error("MongoDB URI not configured in environment variables")
   raise ValueError("MongoDB URI not configured")

try:
    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000
    )
    client.admin.command('ping')
    db = client.UzhavanDB
    users_collection = db.users
    translations_collection = db.translations
    logging.info("MongoDB connection successful")
except Exception as e:
    logging.error(f"MongoDB connection failed: {str(e)}")
    users_collection = None
    translations_collection = None

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, 'videos')
AUDIO_DIR = os.path.join(BASE_DIR, 'audio_output')
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- Model Paths ---
MODEL_DIR = os.path.join(BASE_DIR, 'model')
SOIL_MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_soil (1).pth")
CROP_MODEL_PATH = os.path.join(MODEL_DIR, "crop_model (2) (1).pkl")
IRRIGATION_MODEL_PATH = os.path.join(MODEL_DIR, "irrigation_model (1).pkl")

# Verify model files exist
for path in [SOIL_MODEL_PATH, CROP_MODEL_PATH, IRRIGATION_MODEL_PATH]:
    if not os.path.exists(path):
        logging.warning(f"Model file not found: {path}")

# --- Model Classes and Configurations ---
soil_classes = ["Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil"]
irrigation_classes = ["Very low", "Low", "Moderate", "High", "Very high"]

# Define CustomEfficientNet
class CustomEfficientNet(torch.nn.Module):
    def __init__(self):
        super(CustomEfficientNet, self).__init__() 
        base_model = efficientnet_b0(weights=None)
        self.features = base_model.features
        self.pooling = base_model.avgpool
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Supported Languages ---
SUPPORTED_LANGUAGES = {
    'en': {'gtts': 'en', 'name': 'English'},
    'ta': {'gtts': 'ta', 'name': 'Tamil'},
    'ml': {'gtts': 'ml', 'name': 'Malayalam'},
    'te': {'gtts': 'te', 'name': 'Telugu'},
    'kn': {'gtts': 'kn', 'name': 'Kannada'},
    'hi': {'gtts': 'hi', 'name': 'Hindi'}
}

# --- Gemini AI Integration ---
# --- Gemini AI Integration (ENV based) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_model = None
gemini_chat = None

if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY not found in .env file")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        # ✅ safer model (flash models change often)
        gemini_model = genai.GenerativeModel("models/gemini-1.5-pro")
        gemini_chat = gemini_model.start_chat(history=[])

        logging.info("Gemini API initialized successfully using .env")
        logging.info(f"Gemini model in use: {gemini_model.model_name}")

    except Exception as e:
        logging.error(f"Error initializing Gemini: {str(e)}")
        gemini_model = None
        gemini_chat = None


# gemini_model = None
# gemini_chat = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        gemini_chat = gemini_model.start_chat(history=[])
        logging.info("Gemini API initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing Gemini: {str(e)}")
        gemini_model = None
        gemini_chat = None
else:
    logging.warning("No valid Gemini API key found. Chat functionality will be limited.")

# --- Helper Functions ---
def validate_phone_number(phone_number):
    pattern = r"^[6-9]\d{9}$"
    return bool(re.match(pattern, phone_number))

def clean_text(text):
    return text.replace('*', '').strip() if text else text

def cache_translation(text, source_lang, dest_lang, translated_text):
    if translations_collection and text and translated_text:
        try:
            translations_collection.update_one(
                {'text': text, 'source_lang': source_lang, 'dest_lang': dest_lang},
                {'$set': {'translated_text': translated_text, 'timestamp': datetime.utcnow()}},
                upsert=True
            )
        except Exception as e:
            logging.error(f"Error caching translation: {str(e)}")

def get_cached_translation(text, source_lang, dest_lang):
    if translations_collection and text:
        try:
            cached = translations_collection.find_one({
                'text': text,
                'source_lang': source_lang,
                'dest_lang': dest_lang
            })
            if cached:
                return cached['translated_text']
        except Exception as e:
            logging.error(f"Error retrieving cached translation: {str(e)}")
    return None

def translate_text(text, dest_lang='en'):
    if not text or dest_lang == 'en':
        return text
    if dest_lang not in SUPPORTED_LANGUAGES:
        logging.warning(f"Unsupported language for translation: {dest_lang}")
        return text
    if not gemini_model:
        logging.error("Gemini model not available for translation")
        return text
    # Check cache first
    cached = get_cached_translation(text, 'en', dest_lang)
    if cached:
        return cached
    try:
        prompt = (
            f"Translate the following agricultural term or phrase to {SUPPORTED_LANGUAGES[dest_lang]['name']} "
            "in a natural, accurate, and context-appropriate way for farmers in India. "
            "For measurements like 'tons/ha', use a natural phrasing like 'tons per hectare' in the target language. "
            "Return only the translated text without explanations, brackets, or additional context. "
            f"Text: {text}"
        )
        response = gemini_model.generate_content(prompt)
        if response.parts:
            translated = clean_text("".join(part.text for part in response.parts))
            cache_translation(text, 'en', dest_lang, translated)
            return translated
        return text
    except Exception as e:
        logging.error(f"Translation error ({dest_lang}): {str(e)}")
        return text

def translate_to_english(text, source_lang):
    if source_lang == 'en' or not text:
        return text
    if source_lang not in SUPPORTED_LANGUAGES:
        logging.warning(f"Unsupported source language: {source_lang}")
        return text
    if not gemini_model:
        logging.error("Gemini model not available for translation")
        return text
    cached = get_cached_translation(text, source_lang, 'en')
    if cached:
        return cached
    try:
        prompt = (
            f"Translate the following text from {SUPPORTED_LANGUAGES[source_lang]['name']} to English "
            "in a natural and accurate way, suitable for agricultural contexts in India. "
            "Return only the translated text without explanations or additional context. "
            f"Text: {text}"
        )
        response = gemini_model.generate_content(prompt)
        if response.parts:
            translated = clean_text("".join(part.text for part in response.parts))
            cache_translation(text, source_lang, 'en', translated)
            return translated
        return text
    except Exception as e:
        logging.error(f"Translation error from {source_lang}: {e}")
        return text

def translate_response(data, lang='en'):
    if lang == 'en':
        return data
    try:
        if isinstance(data, dict):
            translated = {}
            for k, v in data.items():
                if k == 'crops' and isinstance(v, list):
                    translated[k] = [
                        {"crop": translate_text(item['crop'], lang), "probability": item['probability']}
                        for item in v
                    ]
                elif k == 'crop' and isinstance(v, str):
                    translated[k] = translate_text(v, lang)
                elif k == 'irrigation' and isinstance(v, str):
                    translated[k] = translate_text(v, lang)
                elif k == 'estimated_yield' and isinstance(v, str):
                    yield_value = v.split()[0] if ' ' in v else v
                    translated[k] = translate_text(f"{yield_value} tons per hectare", lang)
                elif k == 'note' and isinstance(v, str):
                    translated[k] = translate_text(v, lang)
                else:
                    translated[k] = translate_response(v, lang)
            return translated
        elif isinstance(data, list):
            return [translate_response(i, lang) for i in data]
        elif isinstance(data, str):
            return translate_text(data, lang)
        else:
            return str(data)
    except Exception as e:
        logging.error(f"Translation error in response ({lang}): {str(e)}")
        return data

def heuristic_pest_detection(soil_type):
    soil_type = soil_type.lower()
    pest_prefs = {
        "alluvial": ["Locusts", "Aphids", "Whiteflies"],
        "black": ["Armyworm", "Cutworm", "Bollworm"],
        "clay": ["Stem Borer", "Brown Plant Hopper"],
        "red": ["Grasshoppers", "Mites"]
    }
    possible_pests = pest_prefs.get(soil_type, ["Locusts", "Aphids"])
    pest_detection = random.sample(possible_pests, min(2, len(possible_pests)))
    pest_note = "These pests might appear in this soil"
    pest_probability = random.choice(["Low likelihood", "Moderate likelihood", "High likelihood"])
    return pest_detection, pest_note, pest_probability

def heuristic_crop_recommendation(features, soil_type):
    nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall = features
    soil_type = soil_type.lower()

    crop_scores = {
        "rice": 0, "wheat": 0, "maize": 0, "sugarcane": 0, "cotton": 0,
        "groundnut": 0, "barley": 0, "millet": 0, "sorghum": 0, "soybean": 0
    }

    soil_prefs = {
        "alluvial": ["rice", "wheat", "sugarcane"],
        "black": ["cotton", "soybean", "groundnut"],
        "clay": ["rice", "sugarcane"],
        "red": ["groundnut", "millet", "sorghum"]
    }

    for crop in soil_prefs.get(soil_type, []):
        crop_scores[crop] += 2

    if nitrogen > 80:
        for crop in ["rice", "maize", "sugarcane"]:
            crop_scores[crop] += 1
    if phosphorus > 40:
        for crop in ["wheat", "soybean"]:
            crop_scores[crop] += 1
    if potassium > 40:
        for crop in ["cotton", "groundnut"]:
            crop_scores[crop] += 1

    if rainfall > 1000:
        for crop in ["rice", "sugarcane"]:
            crop_scores[crop] += 2
    elif rainfall < 500:
        for crop in ["millet", "sorghum", "barley"]:
            crop_scores[crop] += 2
    if temperature > 25:
        for crop in ["cotton", "maize", "sorghum"]:
            crop_scores[crop] += 1
    if 6.0 <= ph <= 7.5:
        for crop in ["wheat", "soybean", "barley"]:
            crop_scores[crop] += 1

    top_crops = [
        {"crop": crop, "probability": min(0.9, 0.3 + score * 0.1)}
        for crop, score in sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    return top_crops

def check_irrigation_heuristic(crop, soil_type, features):
    crop = str(crop).lower()
    soil_type = soil_type.lower()
    nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall = features

    crop_irrigation = {
        "rice": "high", "sugarcane": "high", "maize": "moderate", "wheat": "moderate",
        "cotton": "moderate", "groundnut": "low", "millet": "low", "barley": "low",
        "sorghum": "low", "soybean": "moderate"
    }
    soil_retention = {
        "clay": "high", "alluvial": "moderate", "black": "high", "red": "low"
    }

    crop_need = crop_irrigation.get(crop, "moderate")
    soil_hold = soil_retention.get(soil_type, "moderate")

    irrigation_score = 0
    if rainfall < 50:
        irrigation_score += 2
    elif rainfall < 200:
        irrigation_score += 1
    if humidity < 60:
        irrigation_score += 1
    if temperature > 30:
        irrigation_score += 1
    if crop_need == "high":
        irrigation_score += 2
    elif crop_need == "moderate":
        irrigation_score += 1
    if soil_hold == "low":
        irrigation_score += 1
    elif soil_hold == "high":
        irrigation_score -= 1

    irrigation_levels = {
        4: "Very high irrigation required",
        3: "High irrigation required",
        2: "Moderate irrigation required",
        1: "Low irrigation required",
        0: "Very low irrigation required"
    }
    return irrigation_levels.get(min(irrigation_score, 4), "Moderate irrigation required")

def check_irrigation(crop, soil_type, features):
    try:
        logging.info(f"Loading irrigation model from {IRRIGATION_MODEL_PATH}")
        irrigation_model = joblib.load(IRRIGATION_MODEL_PATH)
        logging.info("Irrigation model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading irrigation model: {str(e)}")
        return check_irrigation_heuristic(crop, soil_type, features)

    try:
        soil_type_map = {"alluvial": 0, "black": 1, "clay": 2, "red": 3}
        input_data = features + [soil_type_map.get(soil_type.lower(), 0)]
        input_df = pd.DataFrame([input_data], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_type"])
        prediction = irrigation_model.predict(input_df)[0]
        return irrigation_classes[prediction]
    except Exception as e:
        logging.error(f"Error using irrigation model: {str(e)}")
        return check_irrigation_heuristic(crop, soil_type, features)

def estimate_yield(crop, features):
    rainfall = features[6]
    base_yield = {
        "rice": 3.5, "wheat": 2.8, "maize": 2.2, "sugarcane": 6.5, "cotton": 1.5,
        "groundnut": 1.2, "barley": 2.0, "millet": 1.8, "sorghum": 2.0, "soybean": 2.5
    }
    crop = str(crop).lower()
    yield_value = base_yield.get(crop, 2.0)
    if rainfall > 200:
        yield_value *= 1.2
    elif rainfall < 50:
        yield_value *= 0.8
    return float(round(yield_value, 2))

def get_gemini_response(query_en):
    if not query_en or len(query_en.strip()) < 2:
        return "Please provide a more detailed question."
    
    if gemini_chat is None:
        return "Chat service is currently unavailable due to missing API key."

    full_prompt = (
        "You are an AI assistant for farmers in India. For queries about agriculture, farming, crops, soil, or weather, "
        "provide detailed, practical advice in a clear and concise manner, tailored for Indian farmers. "
        "For general knowledge queries, provide accurate and concise answers. "
        "Format responses with 3-5 bullet points where applicable. "
        "Avoid using asterisks (*) for formatting. "
        f"User query: {query_en}"
    )

    try:
        response = gemini_chat.send_message(full_prompt, stream=False)
        if not hasattr(response, 'parts') or not response.parts:
            return "Sorry, I couldn't process that request."
        return clean_text("".join(part.text for part in response.parts if hasattr(part, 'text')).strip())
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        return "Sorry, I encountered an error processing your request."

# --- Routes ---
@app.route('/')
def home():
    return render_template('login.html', languages=SUPPORTED_LANGUAGES)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat_page():
    try:
        return send_from_directory(app.static_folder, 'chatindex.html')
    except FileNotFoundError:
        logging.error("chatindex.html not found in static folder")
        return jsonify({"error": "Chat page not found", "code": "FILE_NOT_FOUND"}), 404

@app.route('/register', methods=['POST'])
def register():
    if not users_collection:
        return jsonify({"error": "Database service unavailable", "code": "DB_UNAVAILABLE"}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided', "code": "NO_DATA"}), 400

    try:
        full_name = data.get('full_name', '').strip()
        email = data.get('email', '').strip().lower()
        phone_number = data.get('phone_number', '').strip()
        password = data.get('password', '').strip()

        if not all([full_name, phone_number, password]):
            return jsonify({'error': 'Full name, phone number, and password are required', "code": "MISSING_FIELDS"}), 400

        if not validate_phone_number(phone_number):
            return jsonify({'error': 'Invalid phone number. Must be a 10-digit Indian mobile number starting with 6-9.', "code": "INVALID_PHONE_NUMBER"}), 400

        if email and not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
            return jsonify({'error': 'Invalid email format', "code": "INVALID_EMAIL"}), 400

        if users_collection.find_one({'phone_number': phone_number}):
            return jsonify({'error': 'Phone number already registered', "code": "PHONE_EXISTS"}), 409
        if email and users_collection.find_one({'email': email}):
            return jsonify({'error': 'Email already registered', "code": "EMAIL_EXISTS"}), 409

        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = {
            'full_name': full_name,
            'email': email or None,
            'phone_number': phone_number,
            'password': hashed_pw,
            'created_at': datetime.utcnow(),
            'verified': False
        }
        
        result = users_collection.insert_one(user)
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user_id': str(result.inserted_id)
        }), 201
    except Exception as e:
        logging.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed', "code": "REGISTRATION_FAILED"}), 500

@app.route('/login', methods=['POST'])
def login():
    if not users_collection:
        return jsonify({"error": "Database service unavailable", "code": "DB_UNAVAILABLE"}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided', "code": "NO_DATA"}), 400

    try:
        email = data.get('email', '').strip().lower()
        phone_number = data.get('phone_number', '').strip()
        password = data.get('password', '').strip()

        if not (email or phone_number) or not password:
            return jsonify({'error': 'Phone number or email and password are required', "code": "MISSING_FIELDS"}), 400

        if phone_number and not validate_phone_number(phone_number):
            return jsonify({'error': 'Invalid phone number. Must be a 10-digit Indian mobile number starting with 6-9.', "code": "INVALID_PHONE_NUMBER"}), 400

        query = {'$or': []}
        if email:
            query['$or'].append({'email': email})
        if phone_number:
            query['$or'].append({'phone_number': phone_number})

        user = users_collection.find_one(query)
        if not user:
            return jsonify({'error': 'Invalid credentials', "code": "INVALID_CREDENTIALS"}), 401

        if not bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return jsonify({'error': 'Invalid credentials', "code": "INVALID_CREDENTIALS"}), 401

        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.utcnow()}}
        )

        return jsonify({
            'success': True,
            'user': {
                'id': str(user['_id']),
                'full_name': user['full_name'],
                'email': user.get('email'),
                'phone_number': user['phone_number']
            }
        })
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed', "code": "LOGIN_FAILED"}), 500

@app.route('/predict_soil', methods=['POST'])
def predict_soil():
    if 'image' not in request.files:
        return jsonify(translate_response({"error": "No image provided", "code": "NO_IMAGE"}, "en")), 400

    lang = request.form.get("language", "en")
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    try:
        logging.info(f"Loading soil model from {SOIL_MODEL_PATH}")
        soil_model = CustomEfficientNet()
        soil_model.load_state_dict(torch.load(SOIL_MODEL_PATH, map_location=torch.device('cpu')))
        soil_model.eval()
        logging.info("Soil model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading soil model: {str(e)}")
        return jsonify(translate_response({"error": "Soil model not loaded", "code": "MODEL_NOT_LOADED"}, lang)), 503

    try:
        image_file = request.files['image']
        img = Image.open(image_file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = soil_model(img_tensor)
            predicted = torch.argmax(output, 1).item()

        soil_type = soil_classes[predicted]
        pest_detection, pest_note, pest_probability = heuristic_pest_detection(soil_type.lower())
        response = {
            "success": True,
            "data": {
                "soil_type": soil_type,
                "pest_detection": pest_detection,
                "pest_note": pest_note,
                "pest_probability": pest_probability
            }
        }
        return jsonify(translate_response(response, lang))
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify(translate_response({
            "error": f"Image processing failed: {str(e)}",
            "code": "IMAGE_PROCESSING_ERROR"
        }, lang)), 500

@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    data = request.get_json()
    lang = data.get("lang", "en") if data else "en"
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    if not data:
        return jsonify(translate_response({"error": "No data provided", "code": "NO_DATA"}, lang)), 400

    try:
        logging.info(f"Loading crop model from {CROP_MODEL_PATH}")
        crop_model = joblib.load(CROP_MODEL_PATH)
        logging.info("Crop model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading crop model: {str(e)}")
        crop_model = None

    try:
        required_fields = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall", "soil_type"]
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        if missing_fields:
            return jsonify(translate_response({
                "error": f"Missing required fields: {missing_fields}",
                "code": "MISSING_FIELDS"
            }, lang)), 400

        features = [
            float(data.get("nitrogen", 0)),
            float(data.get("phosphorus", 0)),
            float(data.get("potassium", 0)),
            float(data.get("temperature", 20)),
            float(data.get("humidity", 50)),
            float(data.get("ph", 6.5)),
            float(data.get("rainfall", 100))
        ]
        
        if any(f < 0 for f in features):
            return jsonify(translate_response({
                "error": "Features cannot be negative",
                "code": "INVALID_FEATURE_VALUE"
            }, lang)), 400
        if not (0 <= features[5] <= 14):
            return jsonify(translate_response({
                "error": "pH must be between 0 and 14",
                "code": "INVALID_PH_VALUE"
            }, lang)), 400
        if features[4] > 100:
            return jsonify(translate_response({
                "error": "Humidity cannot exceed 100%",
                "code": "INVALID_HUMIDITY_VALUE"
            }, lang)), 400
        if features[6] > 5000:
            return jsonify(translate_response({
                "error": "Rainfall cannot exceed 5000 mm",
                "code": "INVALID_RAINFALL_VALUE"
            }, lang)), 400

        soil_type = str(data.get("soil_type", "Alluvial")).capitalize()
        valid_soils = ["Alluvial", "Black", "Clay", "Red"]
        if soil_type not in valid_soils:
            return jsonify(translate_response({
                "error": "Invalid soil type. Must be Alluvial, Black, Clay, or Red.",
                "code": "INVALID_SOIL_TYPE"
            }, lang)), 400

        top_crops = heuristic_crop_recommendation(features, soil_type)
        top_crop = top_crops[0]["crop"] if top_crops else "unknown"
        irrigation_status = check_irrigation(top_crop, soil_type, features)
        estimated_yield = estimate_yield(top_crop, features)
        response = {
            "crop": top_crop,
            "crops": top_crops,
            "irrigation": irrigation_status,
            "estimated_yield": f"{estimated_yield} tons/ha",
            "note": "Using heuristic recommendation" if not crop_model else "Using ML model"
        }
        return jsonify(translate_response(response, lang))
    except Exception as e:
        logging.error(f"Error processing crop recommendation: {str(e)}")
        return jsonify(translate_response({
            "error": f"Error processing crop recommendation: {str(e)}",
            "code": "CROP_RECOMMENDATION_ERROR"
        }, lang)), 500

@app.route('/government_aids', methods=['POST'])
def government_aids():
    data = request.get_json()
    lang = data.get("lang", "en") if data else "en"
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    if not data:
        return jsonify(translate_response({"error": "No data provided", "code": "NO_DATA"}, lang)), 400

    try:
        required_fields = ["state", "land_size"]
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        if missing_fields:
            return jsonify(translate_response({
                "error": f"Missing required fields: {missing_fields}",
                "code": "MISSING_FIELDS"
            }, lang)), 400

        state = str(data.get("state", "")).strip().lower()
        land_size = float(data.get("land_size", 0))

        if not state:
            return jsonify(translate_response({
                "error": "State name is required",
                "code": "MISSING_STATE"
            }, lang)), 400
        if land_size < 0:
            return jsonify(translate_response({
                "error": "Land size cannot be negative",
                "code": "INVALID_LAND_SIZE"
            }, lang)), 400

        for s in all_states:
            if s["state"].lower() == state:
                eligibility_msg = f"{s['eligibility']} You have {land_size} acres of land."
                response = {
                    "success": True,
                    "data": {
                        "state": state.title(),
                        "land_size": float(land_size),
                        "available_schemes": s["available_schemes"],
                        "eligibility": eligibility_msg,
                        "contact": s["contact"]
                    }
                }
                return jsonify(translate_response(response, lang))

        response = {
            "success": True,
            "data": {
                "state": state.title(),
                "land_size": float(land_size),
                "available_schemes": ["No official schemes found for your state"],
                "eligibility": "Unknown",
                "contact": "Visit your local agriculture office for accurate details"
            }
        }
        return jsonify(translate_response(response, lang))
    except Exception as e:
        logging.error(f"Error processing government aids: {str(e)}")
        return jsonify(translate_response({
            "error": str(e),
            "code": "GOVERNMENT_AIDS_ERROR"
        }, lang)), 500

@app.route('/api/grok', methods=['POST'])
def chat():
    data = request.json
    if not data:
        return jsonify(translate_response({
            'error': 'No data provided',
            'code': 'NO_DATA'
        }, "en")), 400

    message = data.get('message', '').strip().lower()
    lang = data.get('language', 'en')
    
    if not message:
        return jsonify(translate_response({
            'error': 'Empty message',
            'code': 'EMPTY_MESSAGE'
        }, lang)), 400
    
    if lang not in SUPPORTED_LANGUAGES:
        lang = 'en'

    if message in ['video', 'show_video', 'crop cultivation video']:
        video_filename = 'crop_cultivation_guide.mp4'
        video_path = os.path.join(VIDEO_DIR, video_filename)
        if os.path.exists(video_path):
            video_url = f'/videos/{video_filename}'
            return jsonify({
                'response': translate_text('Here is the crop cultivation video.', lang),
                'language': lang,
                'video_url': video_url
            })
        return jsonify(translate_response({
            'error': 'Video file not found',
            'code': 'VIDEO_NOT_FOUND'
        }, lang)), 404

    query_en = translate_to_english(message, lang)
    response_en = get_gemini_response(query_en)
    response = translate_text(response_en, lang)
    return jsonify({
        'response': response,
        'language': lang
    })

@app.route('/videos/<filename>')
def serve_video(filename):
    try:
        return send_from_directory(VIDEO_DIR, filename)
    except FileNotFoundError:
        return jsonify(translate_response({
            'error': 'Video file not found',
            'code': 'VIDEO_NOT_FOUND'
        }, "en")), 404

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text', '').strip()
    lang = data.get('language', 'en')
    
    if not text:
        return jsonify(translate_response({
            'error': 'Empty text',
            'code': 'EMPTY_TEXT'
        }, "en")), 400
    
    if lang not in SUPPORTED_LANGUAGES:
        lang = 'en'
    
    gtts_lang = SUPPORTED_LANGUAGES[lang]['gtts']
    timestamp = int(time.time())
    filename = f"tts_{lang}_{timestamp}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    
    try:
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save(filepath)
        return jsonify({
            'audio_url': f'/audio/{filename}',
            'filename': filename
        })
    except Exception as e:
        logging.error(f"TTS error for language {gtts_lang}: {str(e)}")
        return jsonify(translate_response({
            'error': f'Failed to generate speech: {str(e)}',
            'code': 'TTS_ERROR'
        }, "en")), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    try:
        return send_from_directory(AUDIO_DIR, filename)
    except FileNotFoundError:
        return jsonify(translate_response({
            'error': 'Audio file not found',
            'code': 'AUDIO_NOT_FOUND'
        }, "en")), 404

@app.route('/translations.json')
def serve_translations():
    try:
        return send_from_directory('static', 'translations.json')
    except Exception as e:
        return jsonify(translate_response({
            "error": "Translations not found",
            "code": "TRANSLATIONS_NOT_FOUND"
        }, "en")), 404

@app.route('/health', methods=['GET'])
def health():
    status = {
        "api": "running",
        "soil_model": os.path.exists(SOIL_MODEL_PATH),
        "crop_model": os.path.exists(CROP_MODEL_PATH),
        "irrigation_model": os.path.exists(IRRIGATION_MODEL_PATH),
        "gemini": bool(gemini_chat),
        "api_key": bool(key),
        "mongodb": bool(users_collection and translations_collection)
    }
    return jsonify(status)

@app.route('/model_info', methods=['GET'])
def model_info():
    crop_model_info = {
        "loaded": False,
        "path": CROP_MODEL_PATH,
        "exists": os.path.exists(CROP_MODEL_PATH),
        "predict_proba": False,
        "features": "Not loaded",
        "classes": "Not loaded"
    }
    
    try:
        logging.info(f"Loading crop model for model_info from {CROP_MODEL_PATH}")
        crop_model = joblib.load(CROP_MODEL_PATH)
        crop_model_info.update({
            "loaded": True,
            "predict_proba": hasattr(crop_model, 'predict_proba'),
            "features": getattr(crop_model, 'feature_names_in_', "Not available").tolist() if hasattr(crop_model, 'feature_names_in_') else "Not available",
            "classes": getattr(crop_model, 'classes_', "Not available").tolist() if hasattr(crop_model, 'classes_') else "Not available"
        })
        logging.info("Crop model info retrieved successfully")
    except Exception as e:
        logging.error(f"Error loading crop model for model_info: {str(e)}")

    return jsonify({
        "crop_model": crop_model_info,
        "sklearn_version": sklearn.__version__
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 1000))
    app.run(host="0.0.0.0", port=port)

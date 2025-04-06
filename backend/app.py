import psycopg2
from generate_firebase_config import generate_config_file
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from waitress import serve
import torch
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth as firebase_auth
from model import load_model, predict
from database.db_config import get_connection

generate_config_file()


app = Flask(__name__, static_folder="../frontend/dist", static_url_path="/")
CORS(app, supports_credentials=True)

app.secret_key = "your-secret-key"  # Replace with a strong secret key

cred = credentials.Certificate("backend/firebase_config.json")
firebase_admin.initialize_app(cred)

# Load model once at startup
model = load_model()

# --------------------- DATABASE TEST ROUTE ---------------------

@app.route("/api/test_db", methods=["GET"])
def test_db():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM uploads LIMIT 5;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify({"status": "success", "data": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --------------------- SERVE REACT FRONTEND ---------------------

@app.route("/")
def serve_react():
    return send_from_directory(app.static_folder, "index.html")

# --------------------- AUTH ROUTES ---------------------

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    hashed_pw = generate_password_hash(password)

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, email, password, auth_provider) VALUES (%s, %s, %s, %s)",
            (name, email, hashed_pw, 'email')
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Signup successful!"})
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return jsonify({"error": "Email already registered"}), 400
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500

    
@app.route("/google-signup", methods=["POST"])
def google_signup():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")

    if not name or not email:
        return jsonify({"error": "Name and email are required"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Check if user already exists
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()

        if not existing_user:
            cursor.execute(
                "INSERT INTO users (name, email, password, auth_provider) VALUES (%s, %s, %s, %s)",
                (name, email, None, 'google')
            )
            conn.commit()

        cursor.close()
        conn.close()
        return jsonify({"message": "Google signup/login successful", "email": email})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user is None:
            return jsonify({"message": "User not found"}), 404

        user_id, hashed_password = user

        if not hashed_password or not isinstance(hashed_password, str):
            return jsonify({"message": "Invalid Credentials"}), 500  # data corruption check

        if not check_password_hash(hashed_password, password):
            return jsonify({"message": "Incorrect password"}), 401

        session['user_id'] = user_id
        return jsonify({"user": {"id": user_id, "email": email}}), 200

    except Exception as e:
        print(f"Login error: {e}")  # For backend logs
        return jsonify({"message": "Server error", "error": str(e)}), 500

@app.route("/google-login", methods=["POST"])
def google_login():
    try:
        data = request.get_json()
        id_token = data.get("idToken")

        # Verify the token with Firebase Admin SDK
        decoded_token = firebase_auth.verify_id_token(id_token)
        email = decoded_token.get("email")
        uid = decoded_token.get("uid")

        # Connect to your DB and check if user exists
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if not user:
            # If user not in DB, insert them
            cursor.execute(
                "INSERT INTO users (email, google_uid) VALUES (%s, %s) RETURNING id",
                (email, uid)
            )
            user_id = cursor.fetchone()[0]
            conn.commit()
        else:
            user_id = user[0]

        cursor.close()
        conn.close()

        session['user_id'] = user_id
        return jsonify({"user": {"id": user_id, "email": email}}), 200

    except Exception as e:
        print(f"Google login error: {e}")
        return jsonify({"message": "Google login failed", "error": str(e)}), 500

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({"message": "Logged out successfully"})

# --------------------- PREDICTION ROUTE ---------------------

@app.route('/upload', methods=['POST', 'GET'])
def detect():
    if request.method == 'GET':
        return "StegoShield API is running! Use POST request to analyze files."

    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        result, confidence = predict(file, model)

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO results (filename, prediction, confidence, user_id) VALUES (%s, %s, %s, %s)",
            (file.filename, result, confidence, session['user_id'])
        )
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"result": result, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- FRONTEND ROUTING FALLBACK ---------------------

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

# --------------------- RUN SERVER ---------------------

if __name__ == '__main__':
    # Production: use waitress or gunicorn
    # serve(app, host="0.0.0.0", port=5000)
    app.run(debug=True)

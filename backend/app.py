import psycopg2
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from waitress import serve
from backend.model import load_model, predict

app = Flask(__name__)
CORS(app)

conn = psycopg2.connect(
    dbname="payload_detection",
    user="postgres",
    password="root",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

model = load_model()

@app.route('/')
def home():
    return "StegoShield API is running!"

@app.route('/predict', methods=['POST', 'GET'])
def detect():
    if request.method == 'GET':
        return "StegoShield API is running! Use POST request to analyze files."
    file = request.files['file']
    result, confidence = predict(file, model)

    # Save result to database
    cursor.execute("INSERT INTO results (filename, prediction, confidence) VALUES (%s, %s, %s)", 
                   (file.filename, result, confidence))
    conn.commit()

    return jsonify({"result": result, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
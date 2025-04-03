import React, { useState } from "react";

function FileUpload() {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState("");

    const uploadFile = async () => {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch("http://127.0.0.1:5000/predict",{
            method: "POST",
            body: formData,
        });

        const data = await res.json();
        setResult(`Prediction: ${data.result} (Confidence: ${data.confidence}%)`);
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
            <div className="bg-white p-6 rounded-lg shadow-lg">
                <input type="file" className="mb-4" onChange={(e) => setFile(e.target.files[0])} />
                <button onClick={uploadFile} className="px-4 py-2 bg-blue-500 text-white rounded">
                    Upload & Analyze
                </button>
                <p className="mt-4">{result}</p>
            </div>
        </div>
    );
}

export default FileUpload;

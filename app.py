from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Memuat model dan scaler hasil training dari Jupyter Notebook
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil nilai hanya dari input text/number, abaikan dropdown model jika tidak dipakai di python
        # Kita ambil berdasarkan nama field yang dikirim dari form html
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        
        final_features = [np.array(features)]
        
        # Proses normalisasi data menggunakan scaler.pkl
        scaled_features = scaler.transform(final_features)
        
        # Proses prediksi menggunakan model.pkl
        prediction = model.predict(scaled_features)
        
        # Menentukan teks hasil berdasarkan output model (0 atau 1)
        hasil = "Terindikasi Diabetes (1)" if prediction[0] == 1 else "Negatif / Sehat (0)"
        
        return render_template('index.html', prediction_text=f'Hasil Analisis Prediksi: {hasil}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error saat memproses data: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
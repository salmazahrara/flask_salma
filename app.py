from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Memuat model (list berisi Decision Tree & SVC) dan scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)  # Struktur: [Decision Tree, SVC]

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_names = ['Decision Tree', 'SVC']

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Menangkap data sesuai dengan name attribute di HTML
        data = {
            'Pregnancies': int(request.form['pregnancies']),
            'Glucose': int(request.form['glucose']),
            'BloodPressure': int(request.form['blood_pressure']),
            'SkinThickness': int(request.form['skin_thickness']),
            'Insulin': int(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree']),
            'Age': int(request.form['age'])
        }
        
        # Konversi ke DataFrame sesuai kebutuhan model
        df = pd.DataFrame(data, index=[0])
        X = scaler.transform(df)

        # Mengambil model pilihan user dari dropdown
        selected_model_name = request.form['model']
        model_index = model_names.index(selected_model_name)
        clf = model[model_index]
        
        # Jalankan prediksi
        y = clf.predict(X)
        prediction = 'Diabetic' if int(y[0]) == 1 else 'Non-Diabetic'

        return render_template('index.html', model_names=model_names, prediction=prediction)
    
    except Exception as e:
        # Jika ada error, tampilkan pesan error di halaman web
        error_msg = f"Error: {str(e)}"
        return render_template('index.html', model_names=model_names, prediction=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- Load Model dan Fitur ---
model_path = os.environ.get('MODEL_PATH', 'model_churn_pruned.pkl')
features_path = os.environ.get('FEATURES_PATH', 'feature_names.txt')

try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"Feature names loaded from {features_path}, number of features: {len(feature_names)}")
except Exception as e:
    print(f"Error loading feature names: {e}")
    feature_names = []

# --- Fungsi untuk memetakan input ke fitur one-hot ---
def map_input_to_features(input_data):
    """
    Mengonversi input dari form ke DataFrame dengan format one-hot encoded
    sesuai dengan fitur yang digunakan saat pelatihan model.
    """
    # Inisialisasi DataFrame dengan fitur yang benar dan nilai 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)

    # --- Pemetaan Input ke Fitur ---
    # Fitur Numerik
    input_df['tenure'] = float(input_data.get('tenure', 0))
    input_df['MonthlyCharges'] = float(input_data.get('monthly_charges', 0))
    input_df['TotalCharges'] = float(input_data.get('total_charges', 0))

    # Fitur Kategorikal - One-Hot Encoding
    # Contract
    contract_val = input_data.get('contract', 'Month-to-month') # Default ke 'Month-to-month'
    if contract_val == 'Month-to-month':
        input_df['Contract_Month-to-month'] = 1
    elif contract_val == 'One year':
        input_df['Contract_One year'] = 1
    elif contract_val == 'Two year':
        input_df['Contract_Two year'] = 1
    # Jika nilai tidak dikenal, semua fitur kontrak tetap 0 (ini bisa dianggap sebagai kategori 'lainnya' atau kesalahan)

    # PaymentMethod
    payment_val = input_data.get('payment_method', 'Electronic check') # Default
    if payment_val == 'Electronic check':
        input_df['PaymentMethod_Electronic check'] = 1
    elif payment_val == 'Mailed check':
        input_df['PaymentMethod_Mailed check'] = 1
    elif payment_val == 'Bank transfer (automatic)':
        input_df['PaymentMethod_Bank transfer (automatic)'] = 1
    elif payment_val == 'Credit card (automatic)':
        input_df['PaymentMethod_Credit card (automatic)'] = 1

    # TechSupport
    tech_sup_val = input_data.get('tech_support', 'No') # Default
    if tech_sup_val == 'Yes':
        input_df['TechSupport_Yes'] = 1
    elif tech_sup_val == 'No':
        input_df['TechSupport_No'] = 1
    # Tambahkan juga fitur yang tidak disebutkan jika ada (misalnya TechSupport_No internet service)
    # Jika TechSupport_No internet service adalah kategori valid, tambahkan:
    # elif tech_sup_val == 'No internet service':
    #     input_df['TechSupport_No internet service'] = 1

    # OnlineSecurity
    online_sec_val = input_data.get('online_security', 'No') # Default
    if online_sec_val == 'Yes':
        input_df['OnlineSecurity_Yes'] = 1
    elif online_sec_val == 'No':
        input_df['OnlineSecurity_No'] = 1
    # elif online_sec_val == 'No internet service':
    #     input_df['OnlineSecurity_No internet service'] = 1

    # MultipleLines
    multi_line_val = input_data.get('multiple_lines', 'No') # Default
    if multi_line_val == 'Yes':
        input_df['MultipleLines_Yes'] = 1
    elif multi_line_val == 'No':
        input_df['MultipleLines_No'] = 1
    # elif multi_line_val == 'No phone service':
    #     input_df['MultipleLines_No phone service'] = 1

    # DeviceProtection
    dev_prot_val = input_data.get('device_protection', 'No') # Default
    if dev_prot_val == 'Yes':
        input_df['DeviceProtection_Yes'] = 1
    elif dev_prot_val == 'No':
        input_df['DeviceProtection_No'] = 1
    # elif dev_prot_val == 'No internet service':
    #     input_df['DeviceProtection_No internet service'] = 1

    # StreamingTV
    stream_tv_val = input_data.get('streaming_tv', 'No') # Default
    if stream_tv_val == 'Yes':
        input_df['StreamingTV_Yes'] = 1
    elif stream_tv_val == 'No':
        input_df['StreamingTV_No'] = 1
    # elif stream_tv_val == 'No internet service':
    #     input_df['StreamingTV_No internet service'] = 1

    # gender
    gender_val = input_data.get('gender', 'Female') # Default
    if gender_val == 'Male':
        input_df['gender_Male'] = 1
    elif gender_val == 'Female':
        input_df['gender_Female'] = 1 # Biasanya ini adalah baseline, bisa jadi tidak ada kolomnya, tergantung get_dummies

    # SeniorCitizen
    senior_val = input_data.get('senior_citizen', '0') # Harus 0 atau 1
    input_df['SeniorCitizen'] = 1 if senior_val == '1' else 0

    # Partner
    partner_val = input_data.get('partner', 'No') # Default
    if partner_val == 'Yes':
        input_df['Partner_Yes'] = 1
    elif partner_val == 'No':
        input_df['Partner_No'] = 1

    # Dependents
    dep_val = input_data.get('dependents', 'No') # Default
    if dep_val == 'Yes':
        input_df['Dependents_Yes'] = 1
    elif dep_val == 'No':
        input_df['Dependents_No'] = 1

    # PaperlessBilling
    paperless_val = input_data.get('paperless_billing', 'No') # Default
    if paperless_val == 'Yes':
        input_df['PaperlessBilling_Yes'] = 1
    elif paperless_val == 'No':
        input_df['PaperlessBilling_No'] = 1

    # InternetService
    internet_val = input_data.get('internet_service', 'DSL') # Default
    if internet_val == 'DSL':
        input_df['InternetService_DSL'] = 1
    elif internet_val == 'Fiber optic':
        input_df['InternetService_Fiber optic'] = 1
    elif internet_val == 'No':
        input_df['InternetService_No'] = 1

    # OnlineBackup
    backup_val = input_data.get('online_backup', 'No') # Default
    if backup_val == 'Yes':
        input_df['OnlineBackup_Yes'] = 1
    elif backup_val == 'No':
        input_df['OnlineBackup_No'] = 1
    # elif backup_val == 'No internet service':
    #     input_df['OnlineBackup_No internet service'] = 1

    # StreamingMovies
    stream_mov_val = input_data.get('streaming_movies', 'No') # Default
    if stream_mov_val == 'Yes':
        input_df['StreamingMovies_Yes'] = 1
    elif stream_mov_val == 'No':
        input_df['StreamingMovies_No'] = 1
    # elif stream_mov_val == 'No internet service':
    #     input_df['StreamingMovies_No internet service'] = 1

    # tenure_group (jika ada dalam fitur pelatihan, tambahkan logika serupa)
    # ... tambahkan pemetaan untuk fitur lainnya sesuai kebutuhan ...

    # Pastikan kolom sesuai dan dalam urutan yang benar
    # Reindex DataFrame untuk memastikan semua fitur ada dan dalam urutan yang benar
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    return input_df

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not feature_names:
        return jsonify({'error': 'Model or feature names not loaded properly.'}), 500

    try:
        # Ambil data dari form
        input_data = {
            'tenure': request.form['tenure'],
            'monthly_charges': request.form['monthly_charges'],
            'total_charges': request.form['total_charges'],
            'contract': request.form['contract'],
            'payment_method': request.form['payment_method'],
            'tech_support': request.form['tech_support'],
            'online_security': request.form['online_security'],
            'multiple_lines': request.form['multiple_lines'],
            'device_protection': request.form['device_protection'],
            'streaming_tv': request.form['streaming_tv'],
            'gender': request.form['gender'],
            'senior_citizen': request.form['senior_citizen'],
            'partner': request.form['partner'],
            'dependents': request.form['dependents'],
            'paperless_billing': request.form['paperless_billing'],
            'internet_service': request.form['internet_service'],
            'online_backup': request.form['online_backup'],
            'streaming_movies': request.form['streaming_movies'],
            # Tambahkan input lainnya sesuai form
        }

        # Konversi input ke format yang sesuai dengan model
        input_df = map_input_to_features(input_data)

        # Lakukan prediksi
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Konversi hasil prediksi ke string untuk ditampilkan
        result = "Churn (Berisiko Tinggi)" if prediction == 1 else "Tidak Churn (Berisiko Rendah)"
        prob_churn = float(prediction_proba[1]) # Probabilitas kelas 1 (Churn)
        prob_no_churn = float(prediction_proba[0]) # Probabilitas kelas 0 (Tidak Churn)

        return render_template('result.html',
                               result=result,
                               prob_churn=round(prob_churn * 100, 2),
                               prob_no_churn=round(prob_no_churn * 100, 2),
                               input_data=input_data) # Kirim input_data untuk ditampilkan kembali

    except ValueError as e:
        # Jika terjadi kesalahan konversi nilai (misalnya, string ke float)
        return render_template('index.html', error=f"Input Error: {str(e)}. Please enter valid numbers for tenure, monthly charges, and total charges.")

    except Exception as e:
        # Tangani error lainnya
        print(f"Prediction error: {e}") # Log error ke konsol
        return render_template('index.html', error=f"An error occurred during prediction: {str(e)}")


# --- Route untuk API (opsional) ---
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint untuk prediksi (jika dibutuhkan)."""
    if not model or not feature_names:
        return jsonify({'error': 'Model or feature names not loaded properly.'}), 500

    try:
        json_data = request.get_json()
        input_df = map_input_to_features(json_data)

        prediction = int(model.predict(input_df)[0])
        prediction_proba = model.predict_proba(input_df)[0].tolist()

        return jsonify({
            'prediction': prediction,
            'prediction_label': 'Churn' if prediction == 1 else 'Not Churn',
            'prediction_probability': {
                'not_churn': round(prediction_proba[0], 4),
                'churn': round(prediction_proba[1], 4)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Gunakan port dari environment variable (seperti yang disediakan Railway)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) # Set debug=False untuk produksi
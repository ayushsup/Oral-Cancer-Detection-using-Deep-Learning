from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('oral_cancer_model.h5')

# Dummy user data
users = {
    'customer': {'username': 'customer', 'password': '1234', 'role': 'customer'},
    'specialist': {'username': 'specialist', 'password': 'admin', 'role': 'specialist'}
}

predictions_log = []

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    user = users.get(username)
    if user and user['password'] == password:
        session['username'] = username
        session['role'] = user['role']
        if user['role'] == 'customer':
            return redirect(url_for('customer_dashboard'))
        else:
            return redirect(url_for('specialist_dashboard'))
    return render_template('login.html', error='Invalid credentials')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/customer')
def customer_dashboard():
    if 'username' in session and session['role'] == 'customer':
        return render_template('customer_dashboard.html')
    return redirect(url_for('login'))

@app.route('/specialist')
def specialist_dashboard():
    if 'username' in session and session['role'] == 'specialist':
        return render_template('specialist_dashboard.html', logs=predictions_log[::-1])
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session or session['role'] != 'customer':
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)

    # Use softmax result
    predicted_class_index = np.argmax(prediction[0])
    print("Predicted class index:", predicted_class_index)
    prediction_result = 'No Cancer Detected' if predicted_class_index == 0 else 'Cancer Detected'

    print("Raw prediction:", prediction)
    print("Predicted class index:", predicted_class_index)
    print("Prediction result:", prediction_result)

    # Log for specialist
    predictions_log.append({
        'filename': filename,
        'result': prediction_result,
        'user': session.get('username')
    })

    return render_template('customer_dashboard.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)

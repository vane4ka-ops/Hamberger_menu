from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import shutil
from ultralytics import YOLO
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/' # папка для загрузки
app.config['PRED_FOLDER'] = 'runs/detect/predict/' # папка для обработанных изображений
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'} # разрешенные типы файлов

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # очищаем папку для обработанных изображений перед каждым прогнозом
            if os.path.exists('runs/detect'):
                shutil.rmtree('runs/detect')
            # Сохраняем текущее время перед обработкой изображения
            model = YOLO("yolov8s_4_ham.pt")
            start_time = time.time()
            model.predict(file_path, conf=0.4, save=True)
            # Вычисляем затраченное время
            elapsed_time = time.time() - start_time
            return redirect(url_for('show_processed', filename=filename, elapsed_time=elapsed_time))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed_img/<filename>')
def send_processed_file(filename):
    return send_from_directory(app.config['PRED_FOLDER'], filename)

@app.route('/processed/<filename>')
def show_processed(filename):
    elapsed_time = request.args.get('elapsed_time')
    return render_template('processed.html', filename=filename, elapsed_time=elapsed_time)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(port=5000, debug=True)
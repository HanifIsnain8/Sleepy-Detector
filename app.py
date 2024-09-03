from flask import Flask, render_template, Response
import cv2
import time
import pygame
import os

app = Flask(__name__)

face_ref = cv2.CascadeClassifier('face_ref.xml')
eye_ref = cv2.CascadeClassifier('eye_ref.xml')

cap = cv2.VideoCapture(0)

# Variabel global
a = 0  # sleep
b = 0  # active
c = time.time()
ratio = 0  # ratio
danger_detected = False

# Inisialisasi pygame untuk memutar audio
pygame.mixer.init()

# Fungsi untuk mendeteksi wajah pada frame
def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengubah frame ke grayscale
    faces = face_ref.detectMultiScale(optimized_frame)  # Mendeteksi wajah
    return faces

# Fungsi untuk mendeteksi mata pada area wajah yang telah dideteksi
def eye_detection(roi_gray):
    eyes = eye_ref.detectMultiScale(roi_gray)  # Mendeteksi mata
    return eyes

# Fungsi untuk menggambar kotak di sekitar wajah dan mata
def drawer_box(frame, faces):
    global a, b, c, ratio, danger_detected

    for (x, y, w, h) in faces:  
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  
        roi_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)  
        roi_color = frame[y:y + h, x:x + w]  

        eyes = eye_detection(roi_gray)  

        # Mengecek setiap 10 detik untuk memberikan peringatan
        if (time.time() - c) >= 10:
            if a + b > 0:
                ratio = a / (a + b)
            if ratio >= 0.5:
                print("****ALERT*****", a, b, ratio)
                pygame.mixer.music.load('buzz.mp3')  # Memuat file audio
                pygame.mixer.music.play()  # Memainkan audio
                danger_detected = True
            else:
                print("safe", ratio)
                danger_detected = False
            c = time.time()
            a = 0
            b = 0

        # Menghitung frame mata tertutup atau terbuka
        if len(eyes) == 0:
            a += 1
            label = 'sleep'
        else:
            b += 1
            label = 'active'

        # Menampilkan label pada frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Fungsi untuk menghasilkan frame streaming
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            faces = face_detection(frame)  # Mendeteksi wajah dalam frame
            drawer_box(frame, faces)  # Menggambar kotak di sekitar wajah

            ret, buffer = cv2.imencode('.jpg', frame)  # Mengonversi frame ke format JPEG
            frame = buffer.tobytes()  

            yield (b'--frame\r\n' 
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Mengirimkan frame-frame dalam format yang dapat digunakan untuk streaming

@app.route('/')
def index():  
    return render_template('index.html', danger_detected=danger_detected)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

import cv2
import time
import os

face_ref = cv2.CascadeClassifier('face_ref.xml')
eye_ref = cv2.CascadeClassifier('eye_ref.xml')

cap = cv2.VideoCapture(0)

#membuat variable global
a = 0 #sleep
b = 0 #active
c = time.time()

# Fungsi untuk mendeteksi wajah pada frame
def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Mengubah frame ke grayscale
    faces = face_ref.detectMultiScale(optimized_frame) # Mendeteksi wajah
    return faces

# Fungsi untuk mendeteksi mata pada area wajah yang telah dideteksi
def eye_detection(roi_gray):
    eyes = eye_ref.detectMultiScale(roi_gray) # Mendeteksi mata
    return eyes

# Fungsi untuk menggambar kotak di sekitar wajah dan mata
def drawer_box(frame, faces):
    global a, b, c

    for (x, y, w, h) in faces:  #  x,y = koordinat piksel yang menunjukkan posisi sudut kiri atas kotak pembatas.   w = lebar ,tinggi kotak
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Menggambar kotak di sekitar wajah
        roi_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY) # Mengambil area wajah dalam grayscale
        roi_color = frame[y:y + h, x:x + w] # Mengambil area wajah dalam warna asli
        
        eyes = eye_detection(roi_gray) # Mendeteksi mata pada area waja
        
        # Mengecek setiap 15 detik untuk memberikan peringatan
        if (time.time() - c) >= 10:
            if a / (a + b) >= 0.2:
                os.system('buzz.mp3')
                print("****ALERT*****", a, b, a / (a + b))
            else:
                print("safe", a / (a + b))
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


        # Menggambar kotak di sekitar mata yang terdeteksi
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

def close_window():
    cap.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = face_detection(frame)
        drawer_box(frame, faces)
        cv2.imshow('Face and Eye Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()

if __name__ == '__main__':
    main()

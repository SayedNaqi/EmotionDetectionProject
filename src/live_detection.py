"""
============================================================
Emotion Detection - Live Webcam
============================================================
این فایل مدل آموزش‌دیده را لود کرده و به صورت زنده احساسات را تشخیص می‌دهد.
نیازی به اجرای کد آموزش نیست!

نحوه اجرا:
    python live_detection.py

کلیدهای کنترل:
    q یا ESC: خروج از برنامه
============================================================
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import time
import os
import sys

# ==========================================
# 1. تنظیمات اولیه
# ==========================================
# مسیر فایل مدل (همین پوشه)
MODEL_PATH = 'best_efficientnet_emotion.pth'

# بررسی وجود فایل مدل
if not os.path.exists(MODEL_PATH):
    print("="*60)
    print("❌ خطا: فایل مدل پیدا نشد!")
    print(f"   مسیر مورد انتظار: {MODEL_PATH}")
    print("="*60)
    print("\n💡 راه‌حل:")
    print("   1. مطمئن شوید فایل 'best_efficientnet_emotion.pth' در همین پوشه است")
    print("   2. یا مسیر کامل فایل را در متغیر MODEL_PATH وارد کنید")
    print("="*60)
    sys.exit(1)

# انتخاب دستگاه (GPU یا CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# ==========================================
# 2. لود مدل آموزش‌دیده
# ==========================================
print("\n📦 Loading trained model...")
model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded successfully!")

# ==========================================
# 3. لود Face Detector
# ==========================================
print("\n👤 Loading face detector...")
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
print("✅ Face detector ready!")

# ==========================================
# 4. تنظیمات پردازش تصویر
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# نام احساسات
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# رنگ‌های BGR برای هر احساس (برای OpenCV)
emotion_colors = {
    'Angry': (0, 0, 255),      # قرمز
    'Disgust': (0, 128, 128),  # زیتونی
    'Fear': (255, 0, 255),     # بنفش
    'Happy': (0, 255, 255),    # زرد
    'Sad': (255, 128, 0),      # نارنجی
    'Surprise': (255, 255, 0), # فیروزه‌ای
    'Neutral': (0, 255, 0)     # سبز
}

# ایموجی برای هر احساس (اختیاری)
emotion_emojis = {
    'Angry': '😡',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Sad': '😢',
    'Surprise': '😮',
    'Neutral': '😐'
}

# ==========================================
# 5. باز کردن وب‌کم
# ==========================================
print("\n📷 Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam!")
    print("Please check your webcam connection.")
    sys.exit(1)

# تنظیم اندازه فریم برای سرعت بهتر
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Webcam opened!")
print("\n" + "="*60)
print("🎯 Emotion Detection Started!")
print("   Press 'q' or ESC to quit")
print("="*60 + "\n")

# ==========================================
# 6. حلقه اصلی تشخیص
# ==========================================
prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    frame_count += 1
    
    # محاسبه FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # تبدیل به grayscale برای تشخیص چهره
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ✅ خط زیر اضافه شد تا در فریم اول خطا ندهد
    faces = []
    
    # تشخیص چهره‌ها (هر 2 فریم برای سرعت بیشتر)
    if frame_count % 2 == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100)
        )
    
    # پردازش هر چهره
    for (x, y, w, h) in faces:
        # رسم کادر دور چهره
        color = (255, 255, 0)  # زرد پیش‌فرض
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Crop چهره
        face_roi = frame[y:y+h, x:x+w]
        
        # تبدیل به PIL Image
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # اعمال Transform
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        
        # پیش‌بینی
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            _, predicted_idx = torch.max(output, 1)
            
        predicted_emotion = class_names[predicted_idx.item()]
        confidence = probabilities[predicted_idx.item()].item() * 100
        
        # رنگ بر اساس احساس
        color = emotion_colors.get(predicted_emotion, (255, 255, 255))
        
        # رسم کادر با رنگ احساس
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # نمایش نتیجه روی تصویر
        emoji = emotion_emojis.get(predicted_emotion, '')
        label = f"{emoji} {predicted_emotion}: {confidence:.1f}%"
        
        # پس‌زمینه برای متن (برای خوانایی بهتر)
        cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
        cv2.putText(frame, label, (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # نمایش نوار اطمینان
        bar_width = int(w * (confidence / 100))
        cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+15), color, -1)
    
    # نمایش اطلاعات در گوشه تصویر
    cv2.rectangle(frame, (5, 5), (200, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # نمایش فریم
    cv2.imshow('Emotion Detection - Press Q or ESC to Quit', frame)
    
    # بررسی کلید فشرده شده
    key = cv2.waitKey(1) & 0xFF
    
    # خروج با ESC یا q
    if key == 27 or key == ord('q'):
        print("\n🛑 Quitting...")
        break

# ==========================================
# 7. آزادسازی منابع
# ==========================================
cap.release()
cv2.destroyAllWindows()
print("✅ Program finished. Goodbye!")
#!/usr/local/bin/python
#-*- coding:utf-8 -*-
import sys
import zbar
import Image
import cv2
import os
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import shutil
import telepot
from telepot.loop import MessageLoop
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(18,GPIO.OUT)
GPIO.setup(24,GPIO.OUT)
GPIO.output(18,GPIO.HIGH)
GPIO.output(24,GPIO.HIGH)
time.sleep(1)
GPIO.output(18,GPIO.LOW)
GPIO.output(24,GPIO.LOW)

a=0
b=0
bot = telepot.Bot('...')

RESOLUTION = (640, 480)
camera = PiCamera()
camera.resolution = RESOLUTION
rawCapture = PiRGBArray(camera, size=RESOLUTION)

scanner = zbar.ImageScanner()
scanner.parse_config('enable')

subjects = []
with open("names.txt") as file:
    subjects.extend(file.readlines())
with open("ids.txt", 'r') as idfile:
    print idfile.read()
    if idfile.read() != '':
        chat_idg = int(idfile.read())
    else:
        chat_idg = 106103222

def dir_file():
    dirs = os.listdir('training-data')
    global l
    l=0
    for dir_name in dirs:
        l=l+1
    return str(l+1)

def handle(msg):
    global a,i,subjects,chat_idg,c,b
    chat_id = msg['chat']['id']
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type == 'photo' and a==1 or content_type == 'photo' and a==2:
        if a==1:
            global strfile
            strfile=dir_file()
            os.mkdir('training-data/s'+strfile)
        bot.download_file(msg['photo'][-1]['file_id'], 'training-data/s'+strfile+'/{}.jpg'.format(i))
        bot.sendMessage(chat_id, "تصویر دریافت شد!")
        a=2
        i=i+1
    elif content_type == 'text':
        command = msg['text']
        print 'Got command: %s' % command
        if command == '/start':
            bot.sendMessage(chat_id, "به ربات خوش آمدید!")
            return
        if command == '/pass':
            bot.sendMessage(chat_id, "رمز را وارد کنید...")
            b=1
            return
        if b==1 and command == 'raspberrypi':
            with open("ids.txt", "w") as file:
                file.write(str(chat_id))
            chat_idg=chat_id
            bot.sendMessage(chat_id, "رمز درست است:)")
            b=2
            return
        if b==1 and command != 'raspberrypi':
            bot.sendMessage(chat_id, "رمز درست نیست:(")
            return
        if chat_id==chat_idg:
            if command == '/photo':
                bot.sendMessage(chat_id, "تصاویر را ارسال کنید")
                a=1
                i=1
                return
            if command == '/endphoto' and a==2:
                bot.sendMessage(chat_id, "یک نام انتخاب کنید")
                a=3
                return
            if a==3:
                with open("names.txt", "a") as file:
                    file.write(command + "\n")
                bot.sendMessage(chat_id, "نام انتخاب شد:)")
                a=0
                return
            if command == '/cancel' :
                bot.sendMessage(chat_id, "لغو شد!")
                a=0
                return
            if command == '/delphoto':
                with open("names.txt", "w") as file:
                    file.write("" + "\n")
                shutil.rmtree('training-data')
                os.mkdir('training-data')
                bot.sendMessage(chat_id, "اطلاعات حذف شد")
                return
            if command == '/train':
                GPIO.output(18,GPIO.LOW)
                with open("names.txt") as file:
                    datas=file.readlines()
                    subjects = []
                    subjects.extend(datas)
                faces, labels = prepare_training_data("training-data")
                print("Total faces: ", len(faces))
                print("Total labels: ", len(labels))
                bot.sendMessage(chat_id, len(faces))
                face_recognizer = cv2.createLBPHFaceRecognizer()
                face_recognizer.train(faces, np.array(labels))
                GPIO.output(18,GPIO.HIGH)
                return
            if command == '/reboot':
                bot.sendMessage(chat_id, "در حال ریبوت شدن...")
                os.system('sudo shutdown -r now')
                return


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path)
    
    faces = []
    labels = []
    
    for dir_name in dirs:
        
        if not dir_name.startswith("s"):
            continue;
            
        label = int(dir_name.replace("s", ""))
        
        subject_dir_path = data_folder_path + "/" + dir_name
        
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            
            if image_name.startswith("."):
                continue;
            
            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)
            
            cv2.waitKey(100)
            
            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
		try:
		    print(subjects[label])
		except:
		    print("error")
		    if chat_idg is not None:
    		    bot.sendMessage(chat_idg, "خطا")
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

print("Preparing data...")
MessageLoop(bot, handle).run_as_thread()
faces, labels = prepare_training_data("training-data")
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

if len(faces) != 0 :
    face_recognizer = cv2.createLBPHFaceRecognizer()
    face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    if face is not None:
        label, confidence = face_recognizer.predict(face)
        label_text = subjects[label]
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1]-5)
        print(subjects[label])
        if chat_idg is not None:
            bot.sendMessage(chat_idg, subjects[label])
        GPIO.output(24,GPIO.HIGH)
        time.sleep(1)
        GPIO.output(24,GPIO.LOW)
    return img

if chat_idg is not None:
    bot.sendMessage(chat_idg, "ready")
print("Predicting images started!")
GPIO.output(18,GPIO.HIGH)

for filename in camera.capture_continuous('img.jpg'):
    test_img = cv2.imread('img.jpg')

    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY, dstCn=0)
    pil = Image.fromarray(gray)
    width, height = pil.size
    raw = pil.tobytes()
    image = zbar.Image(width, height, 'Y800', raw)
    scanner.scan(image)
    for symbol in image:
        data = symbol.data.decode("utf-8").encode("latin-1")
        print 'decoded', symbol.type, 'symbol', '"%s"' % data
        if chat_idg is not None:
            bot.sendMessage(chat_idg, data)
        GPIO.output(24,GPIO.HIGH)
        time.sleep(1)
        GPIO.output(24,GPIO.LOW)

    predicted_img = predict(test_img)
    print("Predicting...")

    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):
        break



camera.close()
cv2.destroyAllWindows()
GPIO.output(18,GPIO.LOW)
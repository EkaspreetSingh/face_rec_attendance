from flask import Flask, render_template, Response, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cv2 as cv
import os
import face_recognition
import numpy as np


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/engage'
db = SQLAlchemy(app)

class Contacts(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(20), nullable=False)
    msg = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(12))




#capturing opencv face rec
path = r'C:\Users\Asus\Pictures\test'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()

        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            print(name)


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def generate_frames():
    while True:
        success, img = cap.read()
        # img = captureScreen()
        imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
                cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        success, jpg = cv.imencode('.jpg', img)
        buffer = jpg.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')
        # hello_world(name)


@app.route("/2")
def hello_world(n):
    return render_template('tmp.html', name = n)

@app.route("/")
def home_pro():
    return render_template('index.html')

@app.route("/home")
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/contact", methods = ['GET', 'POST'])
def contact():
    if(request.method=='POST'):
        '''Add entry to the database'''
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        entry = Contacts(name=name, msg=message, date=datetime.now(), email=email)
        db.session.add(entry)
        db.session.commit()
    return render_template('contact.html')

@app.route("/att")
def att():
    return render_template('att.html')


@app.route("/video")
def video():
    return Response(generate_frames(),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    with open('Attendance.csv', 'r+') as f:
        data = f.readlines()

    return render_template('list.html', data=data)


if __name__ == "__main__":
    app.run(debug=True)

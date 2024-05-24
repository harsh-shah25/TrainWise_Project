from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
from pymongo import MongoClient
import bcrypt
from flask_cors import CORS
import cv2
import re
import mediapipe as mp
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

# Initialize Mediapipe drawing and pose solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize Flask application
app = Flask(__name__)
CORS(app)
app.secret_key = 'supersecretkey'  # You should use a more secure key

# MongoDB connection setup
client = MongoClient("mongodb+srv://nankanisuneet262:nx082Mb3g6wMk0hP@cluster0.4xepbqd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.get_database('Trainwise') 
users_collection = db.get_collection('users')
exercises_collection = db.get_collection('exercises')

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def count_biceps():
    print('Biceps counting started!')
    cap = cv2.VideoCapture(0)  # 0 for webcam
    counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            try:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except cv2.error as e:
                print("Error processing frame:", e)
                cap.release()
                cv2.destroyAllWindows()
                return f"Error processing frame: {str(e)}", 500

            try:
                landmarks = results.pose_landmarks.landmark

                shoulderLeft = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbowLeft = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wristLeft = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angleLeft = calculate_angle(shoulderLeft, elbowLeft, wristLeft)

                cv2.putText(image, str(angleLeft), tuple(np.multiply(elbowLeft, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 5, 255), 2, cv2.LINE_AA)

                if angleLeft > 150:
                    stage = "down"
                if angleLeft < 60 and stage == 'down':
                    stage = 'up'
                    counter += 1
            except:
                pass

            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return counter


def count_crunches():
    cap = cv2.VideoCapture(0)
    rep_count = 0
    states = {
        1: [130, 180],
        2: [30, 130]
    }
    pattern = '1'
    state = 1
    main_string = "121"
    regex_pattern = ''.join([re.escape(char) + r'+' for char in main_string])

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except cv2.error as e:
                print("Error processing frame:", e)
                cap.release()
                cv2.destroyAllWindows()
                return f"Error processing frame: {str(e)}", 500

            try:
                landmarks = results.pose_landmarks.landmark

                shoulderLeft = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hipLeft = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                kneeLeft = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                left_angle = calculate_angle(shoulderLeft, hipLeft, kneeLeft)

                cv2.putText(image, str(left_angle), tuple(np.multiply(hipLeft, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 5, 255), 2, cv2.LINE_AA)

                if states[1][0] <= left_angle <= states[1][1]:
                    state = '1'
                elif states[2][0] <= left_angle <= states[2][1]:
                    state = '2'

                pattern += state

                if re.fullmatch(regex_pattern, pattern):
                    rep_count += 1
                    pattern = ''

            except:
                pass

            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(rep_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return rep_count


def count_lunges():
    # VIDEO FEED
    cap = cv2.VideoCapture(0)  # 0 is for webcam

    # CURL COUNT
    rep_count = 0
    states = {
                    1:[160,180],
                    2:[110,160],
                    3:[80,110]
                }
    # phase = 'decrease'
    reset = 1
    pattern =''
    main_string = "232"
    regex_pattern = ''.join([re.escape(char) + r'+' for char in main_string])

    ## SETUP MEDIAPIPE INSTANCE
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.9) as pose:


        while cap.isOpened():
            ret, frame = cap.read()
            
            ##Detect stuff and render
#################1
            if not ret:
                break
###########2
            # Recolor image from BGR to RGB
            try:

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make detections
                results = pose.process(image)
                
                #Recolor back to BGR bcuz openCV wants it in BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
###################1
            except cv2.error as e:
                print("4")
                cap.release()
                cv2.destroyAllWindows()
                return f"Error processing frame: {str(e)}", 500
##########2
            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get Co-ordinates for right side
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                
                # Get Co-ordinates for left side
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # Calculate Angle
                
                right_angle = calculate_angle(right_hip, right_knee , right_ankle)
                left_angle = calculate_angle(left_hip, left_knee , left_ankle)
                
                # Visualize
                cv2.putText(image, str(right_angle),
                        tuple(np.multiply(right_knee,[640,480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,5,255), 2, cv2.LINE_AA
                                )
                cv2.putText(image, str(left_angle),
                        tuple(np.multiply(left_knee,[640,480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,5,255), 2, cv2.LINE_AA
                                )
                
    #             state = '1'
                if (states[1][0] <= right_angle <= states[1][1]) and (states[1][0] <= left_angle <= states[1][1]):
                    state = '1'
                elif (states[2][0] <= right_angle <= states[2][1]) and (states[2][0] <= left_angle <= states[2][1]):
                    state = '2'
                elif (states[3][0] <= right_angle <= states[3][1]) and (states[3][0] <= left_angle <= states[3][1]):
                    state = '3'
                    
                
                
                
                reset = 0
                if state == '1':
                    reset = 1
                    count = 0 
                    pattern = ''
                    
                    
                else:
                    pattern += state
    
                    
                if re.fullmatch(regex_pattern, pattern):
                    reset = 1
                    rep_count += 1
                    pattern = ''
                    
                
                
            except:
                pass
            
            cv2.rectangle(image, (0,0), (225,73),(245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS',(15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, str(rep_count),
                        (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE',(65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, 'STAGE',
                        (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
            
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    


        cap.release()
        cv2.destroyAllWindows()
    
    return rep_count

def count_shoulder():
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    rep_count = 0

    states = {
                    1:[0,90],
                    2:[90,120],
                    3:[120,160],
                    4:[160,180]
                }
    
    reset = 0
    pattern =''
    main_string = "23432"
    regex_pattern = ''.join([re.escape(char) + r'+' for char in main_string])
    stage = None

    ## SETUP MEDIAPIPE INSTANCE
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.9) as pose:

        if not cap.isOpened():
            return "Error: Could not open video capture", 500
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make detections
                results = pose.process(image)
                
                #Recolor back to BGR bcuz openCV wants it in BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            except cv2.error as e:
                print("4")
                cap.release()
                cv2.destroyAllWindows()
                return f"Error processing frame: {str(e)}", 500
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get Co-ordinates
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                
                # Calculate Angle
                
                angle = calculate_angle(left_elbow, left_shoulder , left_hip)
                
                # Visualize
                cv2.putText(image, str(angle),
                        tuple(np.multiply(left_shoulder,[640,480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,5,255), 2, cv2.LINE_AA
                                )
                
                if states[1][1] > angle >= states[1][0]:
                    state = '1'
                    stage="down"
                elif states[2][1] >= angle >= states[2][0]:
                    state = '2'
                elif states[3][1] >= angle >= states[3][0]:
                    state = '3'
                elif states[4][1] >= angle >= states[4][0]:
                    state = '4'
                    stage="up"
                
                reset = 0
                if state == '1':
                    reset = 1
                    count = 0
                    pattern = ''
                else:
                    pattern += state

                if re.fullmatch(regex_pattern, pattern):
                    reset = 1
                    rep_count += 1
                    pattern = ''
                    
                
                
            except:
                pass
            
            # Render curl counter
            # setup status box
            cv2.rectangle(image, (0,0), (225,73),(245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS',(15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, str(rep_count),
                        (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE',(65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, stage,
                        (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
            
            # Display the frame
            cv2.imshow('Shoulder Exercise', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return rep_count


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_id = request.form['email_id']
        password = request.form['password']
        user_in_db = users_collection.find_one({'email_id': email_id})
        if user_in_db and bcrypt.checkpw(password.encode('utf-8'), user_in_db['password']):
            session['email_id'] = email_id
            flash('Login successful!')
            return redirect(url_for('home'))  # Redirect to home or the desired page
        else:
            flash('Invalid username or password.')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email_id = request.form['email_id']
        password = request.form['password']
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        users_collection.insert_one({'firstname': first_name, 'last_name': last_name, 'email_id': email_id, 'password': hashed_password})
        flash('Signup successful! Please login.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/<exercise_type>')
def exercise(exercise_type):
    print(exercise_type)
    valid_exercises = ['biceps', 'shoulder', 'crunches', 'lunges']
    if exercise_type in valid_exercises:
        return render_template(f'{exercise_type}.html')
    else:
        return "Exercise not found", 404

# @app.route('/start-exercise/<exercise_type>')
# def start_exercise(exercise_type):
#     exercises = {
#         'biceps': count_biceps,
#         'shoulder': count_shoulder,
#         'crunches': count_crunches,
#         'lunges': count_lunges
#     }
#     if exercise_type in exercises:
#         return process_video(exercises[exercise_type], exercise_type)
#     else:
#         return "Exercise not found", 404

# @app.route('/process_video', methods=['POST'])
# def process_video():
#     if 'email_id' not in session:
#         return "Unauthorized", 401
    
#     exercise_type = request.form.get('exercise_type')
#     if exercise_type not in ['biceps', 'crunches', 'shoulder','lunges']:
#         return "Invalid exercise type", 400

#     if exercise_type == 'biceps':
#         rep_count = count_biceps()
#     elif exercise_type == 'crunches':
#         rep_count = count_crunches()
#     elif exercise_type == 'shoulder':
#         rep_count = count_shoulder()
#     else:
#         rep_count = count_lunges()
    
#     if isinstance(rep_count, tuple):
#         return rep_count  # If rep_count is an error tuple, return it directly

#     email_id = session['email_id']
#     exercise_data = {
#         'email_id': email_id,
#         'exercise': exercise_type,
#         'reps': rep_count,
#         'date': datetime.now()
#     }
#     exercises_collection.insert_one(exercise_data)
    
#     return {'reps': rep_count}

def process_video(exercise_function, exercise_type):
    result = exercise_function()  # Execute the exercise counting function to get the rep count
    if 'email_id' in session:
        email_id = session['email_id']  # Get the user's email ID from the session
        exercises_collection.insert_one({
            'email_id': email_id,
            'exercise_type': exercise_type,
            'rep_count': result,  # Store the rep count
            'timestamp': datetime.datetime.utcnow()  # Store the current timestamp
        })
        flash(f'{exercise_type.capitalize()} exercise completed. Reps: {result}')
    return redirect(url_for('home'))  # Redirect to the home page


# @app.route('/<exercise_type>')
# def exercise(exercise_type):
#     valid_exercises = ['biceps', 'shoulder', 'crunches', 'lunges']
#     if exercise_type in valid_exercises:
#         return render_template(f'{exercise_type}.html')
#     else:
#         return "Exercise not found", 404
    
@app.route('/start-exercise/biceps', methods=['GET'])
def start_biceps():
    return process_video(count_biceps, 'biceps')

@app.route('/start-exercise/crunches', methods=['GET'])
def start_crunches():
    return process_video(count_crunches, 'crunches')

@app.route('/start-exercise/lunges', methods=['GET'])
def start_lunges():
    return process_video(count_lunges, 'lunges')

@app.route('/start-exercise/shoulder', methods=['GET'])
def start_shoulder():
    return process_video(count_shoulder, 'shoulder')

@app.route('/logout')
def logout():
    session.pop('email_id', None)
    flash('Logged out successfully!')
    return redirect(url_for('login'))

@app.route('/workout_history')
def workout_history():
    if 'email_id' in session:
        email_id = session['email_id']
        # Query the database for the user's workout history
        workout_history = list(exercises_collection.find({'email_id': email_id}))
        return render_template('workout_history.html', workout_history=workout_history)
    else:
        flash('You need to log in to view your workout history.')
        return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
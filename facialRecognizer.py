import face_recognition
import cv2
import numpy as np
import dlib

# Initialize a dictionary to store the known face encodings and names
known_faces = {}

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# load the pre-trained model
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

def predict_gender(landmarks):
    """
    Function to predict gender using the face landmarks
    """
    # logic to predict gender using the landmarks
   
    if (landmarks.part(3).y - landmarks.part(0).y) < 0:
        return "Female"
    else:
        return "Male"

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)

    face_landmarks_list = [predictor(rgb_small_frame, dlib.rectangle(left, top, right, bottom)) for (top, right, bottom, left) in face_locations]

    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding, face_landmarks in zip(face_encodings, face_landmarks_list):
        # See if the face is a match for any of the known faces
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)

        name = "Unknown"
        # If no match is found, prompt the user to enter the name of the new face
        if not True in matches:
            name = input("Enter the name of the new face: ")
            known_faces[name] = face_encoding
        else:
            # Find the index of the closest match
            face_distances = face_recognition.face_distance(list(known_faces.values()), face_encoding)
            best_match_index = np.argmin(face_distances)
            name = list(known_faces.keys())[best_match_index]
        # Gender prediction
        gender = predict_gender(face_landmarks)
        name = gender + " - " + name
        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

import cv2
import face_recognition

# Load known faces encodings and names
known_face_encodings = []
known_face_names = []

# Load known faces and their names here
known_person1_image = face_recognition.load_image_file("/Users/honhattri/Downloads/Principal-Component-Analysis/images/person1.jpeg")
known_person2_image = face_recognition.load_image_file("/Users/honhattri/Downloads/Principal-Component-Analysis/images/person2.jpg")
known_person3_image = face_recognition.load_image_file("/Users/honhattri/Downloads/Principal-Component-Analysis/images/person3.jpg")

# Encode faces and append to lists if encoding is available
if face_recognition.face_encodings(known_person1_image):
    known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0]
    known_face_encodings.append(known_person1_encoding)
    known_face_names.append("Tri")
if face_recognition.face_encodings(known_person2_image):
    known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]
    known_face_encodings.append(known_person2_encoding)
    known_face_names.append("Tokuda")
if face_recognition.face_encodings(known_person3_image):
    known_person3_encoding = face_recognition.face_encodings(known_person3_image)[0]
    known_face_encodings.append(known_person3_encoding)
    known_face_names.append("An")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize frame for faster processing

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # Loop through each face in the frame
    for (top, right, bottom, left), current_face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces with a smaller tolerance for accuracy
        matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding, tolerance=0.5)
        name = "Unknown"
        confidence_text = ""

        # Calculate face distances to improve recognition accuracy
        face_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding)
        best_match_index = face_distances.argmin() if face_distances.size > 0 else None
        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = max(0, min(1 - face_distances[best_match_index], 1)) * 100  # Convert to percentage
            confidence_text = f"{confidence:.2f}%"

        # Scale back up face locations since the frame we detected in was scaled
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw label with name and confidence below the face
        cv2.putText(frame, f"{name} ({confidence_text})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the resulting image 
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

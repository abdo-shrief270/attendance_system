import cv2
import dlib
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread
import time

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


def load_known_faces(known_faces_dir):
    encodings = []
    names = []
    encodings_per_person = {}

    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if os.path.isdir(person_dir):
            person_encodings = []
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Detect faces with different scales
                dets = detector(rgb_image, 0)  # Reduced upsampling for faster processing

                for det in dets:
                    shape = shape_predictor(rgb_image, det)
                    face_descriptor = np.array(
                        face_rec_model.compute_face_descriptor(rgb_image, shape, 1))  # Reduced num_jitters
                    person_encodings.append(face_descriptor)
                    encodings.append(face_descriptor)
                    names.append(name)

            if person_encodings:
                encodings_per_person[name] = np.array(person_encodings)

    return np.array(encodings), names, encodings_per_person


def process_frame(frame):
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    return rgb_small_frame


def recognize_face(frame, known_encodings, known_names, encodings_per_person, attendance_log):
    # Process frame at reduced size
    rgb_small_frame = process_frame(frame)

    # Detect faces with reduced upsampling
    dets = detector(rgb_small_frame, 0)

    for det in dets:
        # Scale back the detection coordinates
        scaled_det = dlib.rectangle(
            int(det.left() * 2),
            int(det.top() * 2),
            int(det.right() * 2),
            int(det.bottom() * 2)
        )

        shape = shape_predictor(frame, scaled_det)
        face_descriptor = np.array(face_rec_model.compute_face_descriptor(frame, shape, 1))  # Reduced num_jitters

        # Calculate distances using vectorized operations
        similarities = cosine_similarity(face_descriptor.reshape(1, -1),
                                         known_encodings.reshape(len(known_encodings), -1))[0]

        max_similarity_idx = np.argmax(similarities)
        best_match_name = known_names[max_similarity_idx]

        if best_match_name in encodings_per_person:
            person_encodings = encodings_per_person[best_match_name]
            person_similarities = cosine_similarity(face_descriptor.reshape(1, -1),
                                                    person_encodings.reshape(len(person_encodings), -1))[0]
            confidence = np.mean(np.sort(person_similarities)[-3:]) * 100

            threshold = 0.85

            if confidence > threshold * 100:
                mark_attendance(best_match_name, attendance_log)

                # Draw rectangle and text
                (x, y, w, h) = (scaled_det.left(), scaled_det.top(), scaled_det.width(), scaled_det.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{best_match_name} ({confidence:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Reduced font size
                    (0, 255, 0),
                    1  # Reduced thickness
                )
            else:
                cv2.putText(
                    frame,
                    "Unknown",
                    (scaled_det.left(), scaled_det.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

    return frame


def mark_attendance(name, attendance_log):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if name not in attendance_log:
        attendance_log[name] = current_time
        print(f"Attendance marked for {name} at {current_time}")


def save_attendance(attendance_log):
    df = pd.DataFrame(list(attendance_log.items()), columns=["Name", "Time"])
    df.to_csv("attendance.csv", index=False)
    print("Attendance saved to attendance.csv")


def main():
    print("Loading known faces...")
    known_encodings, known_names, encodings_per_person = load_known_faces('dataset')
    attendance_log = {}

    print("Starting webcam for attendance...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)  # Allow camera to warm up

    fps = FPS().start()

    while True:
        frame = vs.read()
        if frame is None:
            break

        processed_frame = recognize_face(frame, known_encodings, known_names,
                                         encodings_per_person, attendance_log)

        # Calculate and display FPS
        fps.update()
        fps.stop()
        cv2.putText(processed_frame, f"FPS: {fps.fps():.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Smart Attendance System", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    save_attendance(attendance_log)
    vs.stop()
    cv2.destroyAllWindows()


class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = datetime.now()
        return self

    def stop(self):
        self._end = datetime.now()

    def update(self):
        self._num_frames += 1

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._num_frames / self.elapsed()


if __name__ == "__main__":
    main()
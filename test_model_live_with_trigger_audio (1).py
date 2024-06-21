import cv2
import numpy as np
import tensorflow as tf
import pyvirtualcam
import depthai as dai
import time
import pygame  # Import the pygame library

# Initialize pygame mixer
pygame.mixer.init()

# Load the audio file (replace 'your_audio_file.mp3' with the actual path)
audio_file = 'Path to audio'
pygame.mixer.music.load(audio_file)

detection_dict = {0: "Neutral", 1: "Violent"}
'''
json_file = open('Path to model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights("path to weights")'''

#.h5 full model
model = tf.keras.models.load_model("C:\\Users\\ASUS\\PycharmProjects\\pythonProject\\Video_frame\\video_frame15.h5")

print("Loaded model from disk")

#Using Oak 1 camera
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(1280, 720)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
cam.preview.link(xout.input)


sequence_length = 15
sequence = []
predicted_class = ""
prediction = ""
violence_label = []
audio_play_count = 0


with dai.Device(pipeline) as device, pyvirtualcam.Camera(width=1280, height=720, fps=20) as uvc:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    print("UVC running")
    frames = []
    class_label = ""
    output_data = ""

    start_time = time.time()

    while True:

        frame = qRgb.get().getFrame()
        uvc.send(frame)
        frame2 = frame

        frame2 = cv2.resize(frame2, (224, 224))
        img_array = np.array(frame2) / 255.0
        sequence.append(img_array)

        if len(sequence) == sequence_length:

            sequence_frame = np.expand_dims(sequence, axis=0)

    # Make prediction
            prediction = model.predict(sequence_frame)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class = detection_dict[predicted_class_index]
            if predicted_class == "Violent":
                violence_label.append(predicted_class)
            print("Prediction " + predicted_class)

            sequence = []

        if predicted_class == "Violence":
            cv2.putText(frame, predicted_class + " " + str(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, predicted_class + " " + str(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Seconds passed: {int(time.time() - start_time)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        cv2.putText(frame, f"Seconds passed: {int(time.time() - start_time)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 246, 0), 2)
        cv2.imshow("Live Prediction", frame)
        print("Violence label count: " + str(len(violence_label)))
        if time.time() - start_time > 60:  # Check if the time limit (60 seconds) has passed
            print("Time limit reached. Resetting violence label.")
            violence_label = []  # Clear the violence label list
            start_time = time.time()  # Reset the start time

        if len(violence_label) > 5 and audio_play_count < 2:  # Check if violence count exceeds the threshold (10) and audio has not been played three times
            print("Violence threshold exceeded. Playing audio.")
            pygame.mixer.music.play()  # Play the audio
            audio_play_count += 1
            violence_label = []

        audio_play_count = 0

            # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
print("Violence label count: " + str(len(violence_label)))
cv2.destroyAllWindows()

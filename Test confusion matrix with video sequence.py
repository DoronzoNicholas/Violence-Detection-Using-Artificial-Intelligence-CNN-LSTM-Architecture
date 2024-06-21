import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0

        frames.append(frame)

    cap.release()
    return frames


#model = tf.keras.models.load_model("C:/Users/ASUS/PycharmProjects/pythonProject/Video_frame/video_frame15.h5")

json_file = open('C:\\Users\\ASUS\\PycharmProjects\\pythonProject\\sliding_frame\\sliding_frame.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights("C:\\Users\\ASUS\\PycharmProjects\\pythonProject\\sliding_frame\\model2_weights.h5")

#model = tf.keras.models.load_model("C:\\Users\\ASUS\\PycharmProjects\\pythonProject\\Video_frame\\video_frame15.h5")


true_labels = []
predicted_labels = []
class_labels = ['non-violent', 'violent']
count_threshold = 100
detection_dict = {0: "non-violent", 1: "violent"}

violent_folder = "C:\\Users\\ASUS\\Videos\\Testing\\violent"
non_violent_folder = "C:\\Users\\ASUS\\Videos\\Testing\\non-violent"

# Iterate through the folders
for folder, class_label in [(non_violent_folder, "non-violent"), (violent_folder, "violent")]:
    video_files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
    print(f"Folder: {folder}, Class Label: {class_label}")
    print(f"Video files: {video_files}")
    for video_file in video_files:
        video_path = os.path.join(folder, video_file)
        frames = preprocess_video(video_path)
        print("Current Video File: " + str(video_file))
        if len(frames) < 15:
            continue
        violence_predictions = []
        neutral_predictions = []
        for i in range(0, len(frames) - 14):
            video_sequence = frames[i:i+15]
            prediction = model.predict(np.array([video_sequence]))
            print(str(prediction))
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class = detection_dict[predicted_class_index]
            print("Predicted class: " + str(predicted_class))
            print("Current Video File: " + str(video_file))
            if predicted_class == "violent":
                violence_predictions.append(predicted_class)
            else:
                neutral_predictions.append(predicted_class)
        if len(violence_predictions) >= count_threshold:
            predicted_class = "violent"
        else:
            predicted_class = "non-violent"
        frames = []
        true_labels.append(class_label)
        predicted_labels.append(predicted_class)
        print("Predicted classification: " + str(predicted_class) + " True label: " + str(class_label))
        print("Count in violence list: " + str(len(violence_predictions)))
        print("Count in non-violence list: " + str(len(neutral_predictions)))
        violence_predictions.clear()
        neutral_predictions.clear()
        print("Count in violence list: " + str(len(violence_predictions)))
        print("Count in non-violence list: " + str(len(neutral_predictions)))


confusion_mat = confusion_matrix(true_labels, predicted_labels)
class_report = classification_report(true_labels, predicted_labels, target_names=class_labels)

print("Confusion Matrix:")
print(confusion_mat)
print("\nClassification Report:")
print(class_report)


plt.figure(figsize=(6, 4))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["non-violent", "violent"], yticklabels=["non-violent", "violent"])
plt.xlabel("Predicted")
plt.ylabel("True")


plt.savefig("confusion_matrix_slidingframe.png")












import cv2
import time
import os
import shutil
import datetime
import conf
import numpy as np


print("---------------------")
print("| Welcome to PyCCTV |")
print("---------------------")
print("| Version 0.1 Beta  |")
print("---------------------")
print()
print()
print("---------------------")
print("|      Options      |")
print("---------------------")
print(f"1. Headless Mode: {conf.HEADLESS}")
print(f"2. Image Capture Mode: {conf.CAPTURE_IMAGE}")
print(f"3. Video Capture Mode: {conf.CAPTURE_VIDEO}")
print(f"4. Days to Keep Footage: {conf.DAYS_TO_KEEP}")
print(f"5. Capture Cooldown: {conf.COOLDOWN_TIME}")
print()
print("Loading Camera Feed...")


# Load the pre-trained human detection model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the video stream
cap = cv2.VideoCapture('http://192.168.18.171:81/stream')

# Initialize the start time of each human in the frame
human_start_times = {}

time.sleep(2)
print("Camera Feed Loaded!")
print()
while True:

    # Get the current date
    now = datetime.datetime.now()

    # Check the source directory for new images
    for filename in os.listdir(conf.SOURCE_DIR):
        if filename.startswith('human_detection_'):
            # Extract the date from the filename
            parts = filename.split('_')
            date = datetime.datetime.strptime(parts[2], '%Y-%m-%d')

            # Calculate the age of the image in days
            age = (now - date).days

            if age <= conf.DAYS_TO_KEEP:
                # Move the image to the destination directory
                dest_path = os.path.join(conf.DEST_DIR, date.strftime('%Y_%m_%d'))
                if not os.path.exists(dest_path):
                    os.mkdir(dest_path)
                shutil.move(os.path.join(conf.SOURCE_DIR, filename), os.path.join(dest_path, filename))
                print(f"Moved image '{filename}' to '{dest_path}'")
            else:
                # Delete the image if it's too old
                os.remove(os.path.join(conf.SOURCE_DIR, filename))
                print(f"Deleted image '{filename}'")

    # Delete old folders
    if now.minute == 0:
        for dirpath, dirnames, filenames in os.walk(conf.DEST_DIR):
            for dirname in dirnames:
                folder_date = datetime.datetime.strptime(dirname, '%Y_%m_%d')
                folder_age = (now - folder_date).days
                if folder_age > conf.DAYS_TO_KEEP:
                    folder_path = os.path.join(dirpath, dirname)
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder '{folder_path}'")
        
        # Remove empty folders
        for dirpath, dirnames, filenames in os.walk(conf.DEST_DIR, topdown=False):
            for dirname in dirnames:
                folder_path = os.path.join(dirpath, dirname)
                if not os.listdir(folder_path):
                    os.rmdir(folder_path)
                    print(f"Deleted empty folder '{folder_path}'")

    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Exclude the top 20% and bottom 15% of the frame
    height, width = frame.shape[:2]
    roi = frame[int(height * 0.2):int(height * 0.85), :]

    # Check if it is nighttime (between 8:00 pm and 5:00 am)
    now_time = now.time()
    is_night = now_time >= datetime.time(hour=20) or now_time < datetime.time(hour=5)

    # Brighten up if it is nighttime
    if is_night:
        # Convert the color space of the image from RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Apply a bilateral filter to reduce noise while preserving the edges of the objects
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply adaptive thresholding to the filtered image to separate the foreground from the background
        #thresholded = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Convert the binary image back to a color image by applying a colormap (e.g., Hot or Jet) to the thresholded image
        colormap = cv2.applyColorMap(filtered, cv2.COLORMAP_HOT)
        thresholded_colored = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        # Assign the thresholded colored image to the original frame
        frame = thresholded_colored


    # Detect humans in the frame
    start_time = time.time()
    human_boxes, _ = hog.detectMultiScale(roi)

    # Draw bounding boxes around humans that have been detected for longer than MIN_TIME_DETECTED
    for (x, y, w, h) in human_boxes:
        center = (int(x + w/2), int(y + h/2) + int(height * 0.2))
        if center in human_start_times:
            cv2.rectangle(frame, (x, y + int(height * 0.2)), (x + w, y + h + int(height * 0.2)), (0, 255, 0), 2)
        else:
            human_start_times[center] = start_time

    # Save the whole video stream as an image when a human is detected for longer than MIN_TIME_IN_FRAME
    # Check conf file for settings
    
    for center in list(human_start_times):
        if time.time() - human_start_times[center] >= conf.MIN_TIME_IN_FRAME:
            if not conf.cooldown_active:
                if conf.CAPTURE_IMAGE == True:
                    print(f'Human Detected & Captured at: {time.strftime("%Y-%m-%d %H:%M:%S")}')
                    output_file_path = conf.output_file_name + '_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + str(conf.output_file_counter) + '.jpg'
                    cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S'), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.imwrite(output_file_path, frame)

                if conf.CAPTURE_VIDEO == True:
                    print(f'Human Detected & Video Captured at: {time.strftime("%Y-%m-%d %H:%M:%S")}')
                    output_file_path = conf.output_file_name + '_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + str(conf.output_file_counter) + '.mp4'
                    cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S'), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(output_file_path, fourcc, 10, (width, height))
                    start_time = time.time()
                    while time.time() - start_time < 1.5:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                    out.release()

                conf.output_file_counter += 1
                conf.cooldown_active = True
                conf.cooldown_end_time = time.time() + conf.COOLDOWN_TIME
            del human_start_times[center]

    # Check if the cooldown period is over
    if conf.cooldown_active and time.time() >= conf.cooldown_end_time:
        conf.cooldown_active = False

    # Display the video stream if configured
    if conf.HEADLESS == False:
        cv2.imshow('Human detection', frame)
    else:
        pass
    if cv2.waitKey(1) == ord('q'):
        break


# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

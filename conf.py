# Config File
# Change values from true to false (default is recommened)

# choose if the stream will be shown on screen
HEADLESS = False

# choose if an image gets captured if a human is detected
CAPTURE_IMAGE = True
CAPTURE_VIDEO = True

# Set the minimum time a human needs to be in the frame before a picture is taken
MIN_TIME_IN_FRAME = 0.05

# Set the minimum time a human needs to be detected before a bounding box is drawn
MIN_TIME_DETECTED = 0.04

# Set the cooldown time in seconds
COOLDOWN_TIME = 1.0

# Define the output file name and extension
output_file_name = 'human_detection'

# Initialize the output file counter
output_file_counter = 0

# Initialize the cooldown flag
cooldown_active = False
cooldown_end_time = 0

# Set the number of days after which images and folders should be deleted
DAYS_TO_KEEP = 7

# Specify the paths to the source and destination directories
SOURCE_DIR = './'
DEST_DIR = './images'

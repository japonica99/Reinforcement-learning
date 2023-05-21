import os
from PIL import Image

# Define the path of the txt file and the destination folder
txt_file_path = "All_labels.txt"
dest_folder_path_attractive = "attractive/"
dest_folder_path_unattractive = "unattractive/"

# Read the txt file and extract the score values for each line
image_scores = []
with open(txt_file_path, "r") as f:
    for line in f:
        filename, score = line.strip().split(" ")
        image_scores.append((filename, float(score)))

# Sort the scores in descending order
sorted_scores = sorted(image_scores, key=lambda x: x[1], reverse=True)

# Select the top 1500 scores
top_scores = sorted_scores[:1500]
low_scores = sorted_scores[-1500:]

# Extract the image filenames associated with the selected scores
top_filenames = [x[0] for x in top_scores]
low_filenames = [x[0] for x in low_scores]

# Copy the selected image files to a new folder
for filename in top_filenames:
    src_file_path = os.path.join("origin/", filename)
    img = Image.open(src_file_path)
    img_resized = img.resize((512, 512))
    dest_file_path = os.path.join(dest_folder_path_attractive, filename)
    img_resized.save(dest_file_path)


for filename in low_filenames:
    src_file_path = os.path.join("origin/", filename)
    img = Image.open(src_file_path)
    img_resized = img.resize((512, 512))
    dest_file_path = os.path.join(dest_folder_path_unattractive, filename)
    img_resized.save(dest_file_path)

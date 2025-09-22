from PIL import Image
import os

fire_sequences_root = "/home/lagoyal/Full_data/test_sequences/train_sequences/train_fires_final"

fire_sequence_folders = sorted([
    os.path.join(fire_sequences_root, folder)
    for folder in os.listdir(fire_sequences_root)
    if os.path.isdir(os.path.join(fire_sequences_root, folder))
])

for fire_sequence in fire_sequence_folders:

    fire_image_paths = sorted([
        os.path.join(fire_sequence, file)
        for file in os.listdir(fire_sequence)
        if file.endswith(".jpg")
    ])

    print("Path: ", fire_sequence)

    min_pixels = 999999
    max_pixels = 0


    for img_path in fire_image_paths:
        image = Image.open(img_path)
        width, height = image.size
        curr_pixels = width * height

        print("Pixels Image : ", curr_pixels)
        print("Width, Height :", width, height)

        max_pixels = max(max_pixels, curr_pixels)
        min_pixels = min(min_pixels, curr_pixels)

    print("Max Pixels: ", max_pixels)
    print("Min Pixels: ", min_pixels)

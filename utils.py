import numpy as np
import torch
from matplotlib import pyplot as plt
import os
import cv2

def show_stats_np_tensor(np_tensor, title):
    print(title, "mean/std:", np.mean(np_tensor), np.std(np_tensor))
    print(title, "min/max:", np.min(np_tensor), np.max(np_tensor))
    print(title, "shape:", np_tensor.shape)

def show_image(img, title, block=False):
    plt.figure()
    plt.imshow(img[0,0,:,:].detach().cpu().numpy())
    plt.title(title)
    plt.show(block=block)

def get_tensor_stats(tensor, title):
    print("tensor:", title)
    print("mean/std:", torch.mean(tensor), ",", torch.std(tensor))
    print("min/max:", torch.min(tensor), torch.max(tensor))

def dt(tensor):
    return tensor.detach().cpu().numpy()

def get_last_created_model():
    directory = './models'
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
    if len(sorted_files) == 0:
        return None
    else:
        print("starting from:", sorted_files[0])
        return sorted_files[0]

def get_last_generation():
    directory = './'
    files = [os.path.join(directory, f) for f in os.listdir(directory) if (os.path.isfile(os.path.join(directory, f)) and ('.npz' in f))]
    sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
    return sorted_files[0]

def rename_images_in_directory(directory, start_number=5000):
    files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])
    for idx, filename in enumerate(files, start=start_number):
        new_filename = f"Image_{idx}.jpg"
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        os.rename(old_filepath, new_filepath)

def list_files_with_suffix(directory, suffix):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(suffix)]

def create_video_from_images_dir(image_dir, fps=10, target_res=128):
    image_paths = list_files_with_suffix(image_dir, '.png')
    output_path = os.path.join(image_dir, 'clip.mp4')

    # Sort the image paths by name
    sorted_paths = sorted(image_paths, key=lambda x: os.path.basename(x))

    # Read the first image to determine the frame size
    first_image = cv2.imread(sorted_paths[0])
    if target_res:
        first_image = cv2.resize(first_image, (target_res, target_res))

    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 format
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_path in sorted_paths:
        # Read the image
        img = cv2.imread(image_path)
        if target_res:
            img = cv2.resize(img, (target_res, target_res))
        if img is None:
            print(f"Warning: Unable to read {image_path}. Skipping.")
            continue
        # Add the image to the video
        video.write(img)

    # Release the VideoWriter object
    video.release()
    print(f"Video saved to {output_path}")

def create_wide_background_images_from_dir(image_dir, target_res=128, out_dirname='wide_back'):
    image_paths = list_files_with_suffix(image_dir, '.png')

    # Sort the image paths by name
    sorted_paths = sorted(image_paths, key=lambda x: os.path.basename(x))

    for imgpath in sorted_paths:
        image = cv2.imread(imgpath)
        image = cv2.resize(image, (target_res, target_res))
        # Create an 800x300 black background
        background = np.zeros((250, 800, 3), dtype=np.uint8)

        # Get the starting coordinates for centering the image
        x_offset = (background.shape[1] - image.shape[1]) // 2
        y_offset = (background.shape[0] - image.shape[0]) // 2

        # Embed the image in the center of the background
        background[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image

        # Save or display the result
        cv2.imwrite(os.path.join(os.path.dirname(imgpath), out_dirname, os.path.basename(imgpath).replace('.png','_centered.png')), background)


image_dir = r"C:\Development\Playground\stable-diffusion\results\awesome_v6\wide_back"
create_video_from_images_dir(image_dir, target_res=None)
# create_wide_background_images_from_dir(image_dir)
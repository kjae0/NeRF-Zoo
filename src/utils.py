import os
import cv2
import torch
import numpy as np
import imageio

def save_checkpoints(ckpt_path, ckpt, epoch, train_loss, val_loss=None):
    torch.save({
        'epoch': epoch,
        'checkpoint': ckpt,
        'train_loss': train_loss,
        'val_loss': val_loss
    }, ckpt_path)

def save_images(save_dir, images, ground_truth=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for i, image in enumerate(images):
        image = np.array(image)
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Convert pixel values from [0, 1] to [0, 255]
        image = (image * 255).astype(np.uint8)
        
        # Save the image using imageio
        imageio.imwrite(os.path.join(save_dir, f"{i}.png"), image)
    
    if ground_truth:
        for i, image in enumerate(ground_truth):
            image = np.array(image)
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # Convert pixel values from [0, 1] to [0, 255]
            image = (image * 255).astype(np.uint8)
            
            # Save the image using imageio
            imageio.imwrite(os.path.join(save_dir, f"gt_{i}.png"), image)

def save_gif(save_dir, images):
    img_for_gif = []
    for i, image in enumerate(images):
        image = np.array(image)
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Convert pixel values from [0, 1] to [0, 255]
        image = (image * 255).astype(np.uint8)
        img_for_gif.append(image)
        
        # Save the image using imageio
    imageio.mimsave(save_dir, img_for_gif, fps=5)
        
def images_to_video(images, output_video_path, fps=30):
        height, width = images[0].shape[:2]

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change this to other codecs, like 'XVID' for .avi
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Loop through all images and add them to the video
        for image in images:
            image = (np.array(image) * 255).astype(np.uint8)
            
            # Ensure image is in correct color format (BGR for OpenCV)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            video.write(image)

        # Release the video writer
        video.release()

import cv2
import torch
import numpy as np

def save_checkpoints(ckpt_path, ckpt, epoch, train_loss, val_loss=None):
    torch.save({
        'epoch': epoch,
        'checkpoint': ckpt,
        'train_loss': train_loss,
        'val_loss': val_loss
    }, ckpt_path)


# def images_to_video(images, out_path=None):
    
#     if out_path:
        
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
        
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import time

def load_mask_from_file(mask_file):
    with open(mask_file, 'r') as f:
        mask_size = int(f.readline())
        mask_values = [[float(value) for value in line.split()] for line in f.readlines()]
    mask = torch.tensor(mask_values, dtype=torch.float32)
    return mask

def convolve_channel(channel, mask):
    # Convert channel to torch tensor and normalize
    channel_tensor = transforms.ToTensor()(channel).unsqueeze(0)
    mask_tensor = mask.unsqueeze(0).unsqueeze(0)

    # Apply convolution using built-in PyTorch functions
    convolved_channel = torch.nn.functional.conv2d(channel_tensor, mask_tensor, padding=mask.shape[0] // 2)

    return convolved_channel.squeeze(0).squeeze(0).numpy()

def main(input_folder, output_folder, mask_file, batch_size):
    # Load the convolution mask from file
    mask = load_mask_from_file(mask_file)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all files in the input folder
    image_files = os.listdir(input_folder)

    total_time = 0  # Variable to store the total time

    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]

        for image_file in batch_files:
            # Read the input image
            input_image_path = os.path.join(input_folder, image_file)
            output_image_path = os.path.join(output_folder, image_file)
            input_image = Image.open(input_image_path)
            if input_image is None:
                print("Error: Unable to read input image:", input_image_path)
                continue

            # Start the timer
            start_time = time.time()

            # Split the image into channels
            channels = input_image.split()

            # End the timer
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time

            # print the time with 8 floating point precision
            print(f"Time taken to split the image into channels: {elapsed_time:.8f} seconds")
            # Convolve each channel
            convolved_channels = [convolve_channel(channel, mask) for channel in channels]

            # Sum the convolved channels
            output_image = sum(convolved_channels)

            # Normalize the output image to [0, 255]
            output_image -= output_image.min()
            output_image /= output_image.max()
            output_image *= 255
            output_image = output_image.astype('uint8')

            # Convert to PIL image
            output_image = Image.fromarray(output_image, 'L')

            # Save the output image
            output_image.save(output_image_path)



if __name__ == "__main__":
    input_folder = "in1"
    output_folder = "output"
    mask_file = "kernel.txt"
    batch_size = 5 

    # Run the main function
    main(input_folder, output_folder, mask_file, batch_size)

import os
import torch
import tifffile as tif
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
#from torchvision import transforms
import matplotlib.pyplot as plt

class CellDivisionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, device=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.device = torch.device(device if device else "cpu")
        self.img_labels_bin = self.img_labels.iloc[:, 1].apply(lambda x: 1 if x in [3, 7, 8] else 0)


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(str(self.img_dir), str(self.img_labels.iloc[idx, 0]))

        # define the label
        # label = self.img_labels.iloc[idx, 1]
        label = self.img_labels_bin[idx]

        # Use tifffile to read the stacked image directly
        image_stack = tif.imread(img_path)

        # If the image has a different dtype (e.g., uint16), you might want to convert it
        # For example, converting to float and scaling to [0, 1] if necessary
        #image_stack_tensor = image_stack_tensor.float() / image_stack_tensor.max()

        image_stack = image_stack.astype(np.float32) / np.iinfo(image_stack.dtype).max

        # If the image is not already in a torch tensor, convert it
        # Assuming image_stack is a numpy array of shape [num_frames, H, W] for grayscale images
        image_stack_tensor = torch.from_numpy(image_stack).to(self.device)

        # Trim the tensor to have a fixed depth of 20 if it has more
        if image_stack_tensor.shape[0] > 20:
            image_stack_tensor = image_stack_tensor[:20, :, :]


        # Add a channel dimension to the video tensor
        image_stack_tensor = image_stack_tensor.unsqueeze(0)  # This adds the channel dimension at position 0

        # Convert label to a tensor (if it's not already one)
        label = torch.tensor(label)

        # Apply any transforms here. Note: You'll need to modify or ensure your transforms
        # can handle 3D data if they're meant for 2D images.
        if self.transform:
            # Apply transform to each frame individually or modify your transform to handle 3D data
            # This is a placeholder; actual implementation will depend on your transform
            pass

        return image_stack_tensor, label


    def show_stack(self, idx):
        """
        Display a stack of images from a dataset.

        This function takes the first sample from the provided dataset, assuming it is a tuple
        where the first element is a stack of images (as a torch tensor) and the second element is its label.
        It then plots the images in a 5x5 grid, displaying up to 25 images from the stack.

        Parameters:
        - dataset: A dataset object that allows indexing to retrieve a sample (image stack and label).

        Returns:
        None. This function directly displays the plotted images.
        """
        img, label = self[idx]  # Extract the first sample (image stack and label)
        # Set up the figure and axes for plotting
        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))

        # Flatten the axes array for easy iteration
        axes_flat = axes.flatten()

        for i, ax in enumerate(axes_flat):
            if i < len(img):
                # reshape removing the channel dimension
                img = img.reshape(20, 75, 75)
                ax.imshow(img.cpu().numpy()[i], cmap="gray")  # Assuming img is a torch tensor
                ax.set_title(f"Frame {i+1}")
                ax.axis("off")
            else:
                ax.axis("off")  # Hide unused subplots

        plt.tight_layout()
        plt.show()

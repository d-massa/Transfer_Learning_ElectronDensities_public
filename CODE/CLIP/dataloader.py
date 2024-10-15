import torch
from torch.utils.data import Subset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
from pathlib import Path
import os 
from torchvision.transforms import v2
import numpy as np

class MaterialDataset(Dataset):
    def __init__(self, df, image_folder, processor,target, images_prefix, normalize, target_mean, target_std, transform=None):
        self.df = df
        self.image_folder = image_folder
        self.processor = processor
        self.target=target
        self.transform = transform
        self.images_prefix=images_prefix
        self.normalize=normalize
        self.target_mean=target_mean
        self.target_std=target_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_folder, f"{self.images_prefix}_{row.name}.png")
        image = Image.open(image_path).convert("RGB")
        text = row['Description']
        target = torch.tensor(row[self.target], dtype=torch.float)

        if self.normalize:
            image_array = np.array(image)  # Convert the image to a NumPy array
            image = image_array / 255.0  # Normalize the pixel values to the range [0, 1]
            target = (target - self.target_mean) / self.target_std

        if self.transform:
            image = self.transform(image)

        # Process the image and text together
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding='max_length', truncation=True)

        return inputs, target

def dataprep(paths,holdout_ratio, seedd, target, augm, clip_processor,images_prefix, normalize, nrows):

    image_transforms = v2.Compose([
    v2.RandomResizedCrop(size=(186, 186), antialias=True),
    v2.RandomHorizontalFlip(p=0.2)])

    image_folder = Path(paths[0])
    csv_file_props = paths[1]
    csv_file_Descr = paths[2]

    df_props = pd.read_csv(csv_file_props, index_col=0)  # y
    df_Descr = pd.read_csv(csv_file_Descr, index_col=0)  # X_text

    if nrows!=0:
        df_props = pd.read_csv(csv_file_props, index_col=0,nrows=nrows)  # y
        df_Descr = pd.read_csv(csv_file_Descr, index_col=0,nrows=nrows)  # X_text

    if normalize: 
        target_mean = df_props[target].mean()
        target_std = df_props[target].std()        

    df_combined = df_props.join(df_Descr, how='inner')

    if augm: 
        dataset = MaterialDataset(df_combined, image_folder, clip_processor, target, images_prefix, normalize, target_mean, target_std, transform=image_transforms)
    else:
        dataset = MaterialDataset(df_combined, image_folder, clip_processor, target, images_prefix, normalize, target_mean, target_std, transform=None)
  

    # Split the dataset into training+validation set and holdout set
    train_val_indices, holdout_indices = train_test_split(
        range(len(dataset)),
        test_size=holdout_ratio,
        random_state=seedd,
        shuffle=True
    )

    # Create subsets for training+validation and holdout sets
    train_val_dataset = Subset(dataset, train_val_indices)
    holdout_dataset = Subset(dataset, holdout_indices)

    return train_val_dataset, holdout_dataset

# def dataprep(paths,holdout_ratio, seedd, target, nrows):

#     image_folder = Path(paths[0])
#     csv_file_props = paths[1]
#     csv_file_Descr = paths[2]

#     df_props = pd.read_csv(csv_file_props, index_col=0)  # y
#     df_Descr = pd.read_csv(csv_file_Descr, index_col=0)  # X_text

#     if nrows!=0:
#         df_props = pd.read_csv(csv_file_props, index_col=0,nrows=nrows)  # y
#         df_Descr = pd.read_csv(csv_file_Descr, index_col=0,nrows=nrows)  # X_text

#     df_combined = df_props.join(df_Descr, how='inner')

#     dataset = MaterialDataset(df_combined, image_folder, clip_processor, target, transform=None)
#     print(f'dataset is {len(dataset)} long')

#     # Define the ratio for the holdout set
#     # 20% of the data as holdout

#     # Split the dataset into training+validation set and holdout set
#     train_val_indices, holdout_indices = train_test_split(
#         range(len(dataset)),
#         test_size=holdout_ratio,
#         random_state=seedd,
#         shuffle=True
#     )

#     # Create subsets for training+validation and holdout sets
#     train_val_dataset = Subset(dataset, train_val_indices)
#     holdout_dataset = Subset(dataset, holdout_indices)

#     return train_val_dataset, holdout_dataset

# class AugmentedDataset(Dataset):
#     def __init__(self, original_dataset, augmentations):
#         self.original_dataset = original_dataset
#         self.augmentations = augmentations

#     def __len__(self):
#         # Number of original samples multiplied by the number of augmentations
#         return len(self.original_dataset) * len(self.augmentations)

#     def __getitem__(self, idx):
#         original_idx = idx // len(self.augmentations)
#         aug_idx = idx % len(self.augmentations)
        
#         # Get the original image, text, and target
#         inputs, target = self.original_dataset[original_idx]
#         image = inputs['images'][0].convert("RGB")  # Assuming `inputs` contains a PIL Image
#         text = inputs['text'][0]
        
#         # Apply the augmentation
#         augmented_image = self.augmentations[aug_idx](image)
        
#         # Create a new set of inputs with the augmented image
#         augmented_inputs = self.original_dataset.processor(text=[text], images=[augmented_image], return_tensors="pt", padding='max_length', truncation=True)
        
#         return augmented_inputs, target


# def dataprep_augm(paths, holdout_ratio, seedd, target, nrows):
    
#     image_folder = Path(paths[0])
#     csv_file_props = paths[1]
#     csv_file_Descr = paths[2]

#     df_props = pd.read_csv(csv_file_props, index_col=0)  # y
#     df_Descr = pd.read_csv(csv_file_Descr, index_col=0)  # X_text

#     if nrows != 0:
#         df_props = pd.read_csv(csv_file_props, index_col=0, nrows=nrows)  # y
#         df_Descr = pd.read_csv(csv_file_Descr, index_col=0, nrows=nrows)  # X_text

#     df_combined = df_props.join(df_Descr, how='inner')

#     # Define the augmentations
#     image_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(30),
#         transforms.ToTensor()
#     ])

#     # Create the original dataset
#     dataset = MaterialDataset(df_combined, image_folder, clip_processor, target)
#     print(f'Dataset length without augmentation: {len(dataset)}')

#     # Create the augmented dataset
#     augmented_dataset = AugmentedDataset(dataset, [image_transforms])
#     print(f'Dataset length with augmentation: {len(augmented_dataset)}')

#     # Split the dataset into training+validation set and holdout set
#     train_val_indices, holdout_indices = train_test_split(
#         range(len(augmented_dataset)),
#         test_size=holdout_ratio,
#         random_state=seedd,
#         shuffle=True
#     )

#     # Create subsets for training+validation and holdout sets
#     train_val_dataset = Subset(augmented_dataset, train_val_indices)
#     holdout_dataset = Subset(augmented_dataset, holdout_indices)

#     return train_val_dataset, holdout_dataset
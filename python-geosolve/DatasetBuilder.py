import torch
from torch.utils.data import Dataset, DataLoader
import zipfile as zp
import numpy as np
import pandas as pd
import os
import io
from PIL import Image
import torchvision.transforms as transforms

class GeoCoordDataset(torch.utils.data.Dataset):
    MAIN_ZIP = "dataset/archive.zip"
    INTERNAL_DATA = "dataset/"
    CSV_PATH = INTERNAL_DATA + "coords.csv"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    BUFFER_SIZE = 1024 

    def __init__(self, main_zip_path=None, internal_data_path=None, csv_path=None,
                 img_size=None, batch_size=None, buffer_size=None, transform=None,
                 cache_images_in_memory=True):

        self.main_zip_path = main_zip_path if main_zip_path is not None else self.MAIN_ZIP
        self.internal_data_path = internal_data_path if internal_data_path is not None else self.INTERNAL_DATA
        self.csv_path = csv_path if csv_path is not None else self.CSV_PATH
        self.img_size = img_size if img_size is not None else self.IMG_SIZE
        self.batch_size = batch_size if batch_size is not None else self.BATCH_SIZE
        self.buffer_size = buffer_size if buffer_size is not None else self.BUFFER_SIZE
        self.cache_images_in_memory = cache_images_in_memory

        self.image_label_list = self._load_image_labels()
        self.image_byte_cache = {} # Initialize cache

        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        if self.cache_images_in_memory:
            print("Pre-reading all image bytes into memory for caching...")
            with zp.ZipFile(self.main_zip_path, 'r') as zf:
                for img_file_name_in_zip, _ in self.image_label_list:
                    with zf.open(img_file_name_in_zip) as imgFile:
                        self.image_byte_cache[img_file_name_in_zip] = imgFile.read()
            print(f"Finished pre-reading {len(self.image_byte_cache)} images.")

    def _load_image_labels(self):
        """Loads image filenames and their corresponding coordinates from the CSV."""
        image_label_list = []
        with zp.ZipFile(self.main_zip_path, 'r') as zf:
            with zf.open(self.csv_path) as csvBytes:
                data = io.StringIO(csvBytes.read().decode("utf-8"))
                setOfCoords = pd.read_csv(data, header=None)
                setOfCoords.columns = ["longitude", "latitude"]

        for index, row in setOfCoords.iterrows():
            key = self.internal_data_path + f"{index}.png"
            value = np.array([row["longitude"], row["latitude"]], dtype=np.float32)
            image_label_list.append((key, value))
        return image_label_list

    def __len__(self):
        return len(self.image_label_list)

    def __getitem__(self, idx):
        img_file_name_in_zip, coords = self.image_label_list[idx]

        if self.cache_images_in_memory and img_file_name_in_zip in self.image_byte_cache:
            img_bytes = self.image_byte_cache[img_file_name_in_zip]
        else:
            with zp.ZipFile(self.main_zip_path, 'r') as zf:
                with zf.open(img_file_name_in_zip) as imgFile:
                    img_bytes = imgFile.read()
                    if self.cache_images_in_memory: # Cache if not already pre-read
                         self.image_byte_cache[img_file_name_in_zip] = img_bytes

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB") # Ensure 3 channels

        if self.transform:
            img = self.transform(img)

        coords_tensor = torch.tensor(coords, dtype=torch.float32)

        return img, coords_tensor

    def get_dataloader(self, shuffle=True, num_workers=0):
        dataloader_args = {
            'batch_size': self.batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
        }

        if num_workers > 0 and torch.cuda.is_available():
            dataloader_args['pin_memory'] = True
        elif num_workers == 0:
            pass

        return DataLoader(self, **dataloader_args)
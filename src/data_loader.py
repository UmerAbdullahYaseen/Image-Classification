import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from src.utils import get_transform, get_classes
from PIL import Image


class CustomDataset(Dataset):

    def __init__(self, root_dir, split='train',BATCH_SIZE=64):

        self.root_dir = root_dir
        self.transform = get_transform()
        self.BATCH_SIZE = BATCH_SIZE
        self.classes = sorted(get_classes(self.root_dir))

        all_files = []
        all_labels = []
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, class_name)
            files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
            all_files.extend(files)
            all_labels.extend([label] * len(files))

        train_files, test_files, train_labels, test_labels = train_test_split(
            all_files, all_labels, test_size=0.2, random_state=42
        )
        valid_files, test_files, valid_labels, test_labels = train_test_split(
            test_files, test_labels, test_size=0.5, random_state=42
        )

        if split == 'train':
            self.data = train_files
            self.labels = train_labels
        elif split == 'valid':
            self.data = valid_files
            self.labels = valid_labels
        elif split == 'test':
            self.data = test_files
            self.labels = test_labels
        else:
            raise ValueError("Invalid split, Use 'train', 'valid' or 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label

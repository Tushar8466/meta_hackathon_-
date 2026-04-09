import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CropDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Step 1: Get valid class folders only
        self.classes = []
        for d in os.listdir(root_dir):
            full_path = os.path.join(root_dir, d)

            if d.startswith("."):  # skip hidden files like .DS_Store
                continue

            if not os.path.isdir(full_path):
                continue

            self.classes.append(d)

        self.classes = sorted(self.classes)

        # Map class to index
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Step 2: Load image paths safely
        valid_extensions = (".jpg", ".jpeg", ".png")

        for label in self.classes:
            folder = os.path.join(root_dir, label)

            if not os.path.isdir(folder):
                continue

            for img in os.listdir(folder):

                # Skip hidden/system files
                if img.startswith("."):
                    continue

                # Only allow image files
                if not img.lower().endswith(valid_extensions):
                    continue

                img_path = os.path.join(folder, img)

                if not os.path.isfile(img_path):
                    continue

                self.data.append((img_path, label))

        print("✅ Classes:", self.classes)
        print("📊 Total images loaded:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # Skip corrupted image and move to next
            return self.__getitem__((idx + 1) % len(self.data))

        label = self.class_to_idx[label]
        return self.transform(image), label
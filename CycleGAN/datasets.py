import glob
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode="train"):
        self.root = root
        self.transform = transform  # Remove the Compose here
        self.files_A = os.path.join(root, mode, "trainA/*")
        self.files_B = os.path.join(root, mode, "trainB/*")

        self.list_A = glob.glob(self.files_A)
        self.list_B = glob.glob(self.files_B)

    def __len__(self):
        return max(len(self.list_A), len(self.list_B))

    def __getitem__(self, index):
        img_pathA = self.list_A[index % len(self.list_A)]
        img_pathB = random.choice(self.list_B)

        item_A = Image.open(img_pathA).convert("RGB")
        item_B = Image.open(img_pathB).convert("RGB")

        if self.transform:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)

        return {"A": item_A, "B": item_B}


# if __name__ == "__main__":
#     from torch.utils.data import DataLoader

#     root = "data"
#     # Define transforms as Compose here
#     transform = transforms.Compose(
#         [transforms.Resize(256, Image.BILINEAR), transforms.ToTensor()]
#     )

#     dataloader = DataLoader(
#         ImageDataset(root, transform, "train"),
#         batch_size=1,
#         shuffle=True,
#         num_workers=1,
#     )

#     for i, data in enumerate(dataloader):
#         print(i, data["A"].shape, data["B"].shape)
#         if i == 10:
#             break

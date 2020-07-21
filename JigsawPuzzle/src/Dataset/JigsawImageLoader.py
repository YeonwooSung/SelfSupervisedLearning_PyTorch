import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image


class DataLoader(data.Dataset):
    def __init__(self, data_path, classes=1000):
        self.img_paths = []
        for cls in os.listdir(data_path):
            for img in os.listdir(os.path.join(data_path, cls)):
                self.img_paths.append(os.path.join(data_path, cls, img))

        self.permutations = np.load(f'permutations/permutations_{classes}.npy')

        self.image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        
        if np.random.rand() < 0.30:
            img = img.convert('LA').convert('RGB')
            
        if img.size[0] != 255:
            img = self.image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            #norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            #tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data)

        return data, int(order), tiles

    def __len__(self):
        return len(self.img_paths)


def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')

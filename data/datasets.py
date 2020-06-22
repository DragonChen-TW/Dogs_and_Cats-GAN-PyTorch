import os
#
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class DogCatData(data.Dataset):
    def __init__(self, root='D:/data/dogs_cats'):
        '''
        Get Image Path.
        '''
        folder = os.path.join(root, 'train')
        imgs = [os.path.join(folder, img).replace('\\', '/') \
                for img in os.listdir(folder) if 'cat' in img]

        # train: data/train/cat.10004.jpg
        imgs = sorted(imgs, key=lambda img: int(img.split('.')[-2]))
        imgs_num = len(imgs)
        imgs = imgs[:int(imgs_num * 0.5)]

        # split into train, val
        self.imgs = imgs
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.transforms = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        '''
        return one image's data
        if in test dataset, return image's id
        '''
        img_path = self.imgs[index]

        data = Image.open(img_path)
        data = self.transforms(data)

        label = 1

        return data, label

    def __len__(self):
        return len(self.imgs)

def SVHN(root='D:\data\SVHN'):
    transform = transforms.Compose([
        transforms.Resize(80),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return datasets.SVHN(root, split='test', transform=transform)

if __name__ == '__main__':
    dataset_train = SVHN()

    train_data = DataLoader(dataset_train,
        shuffle=True, batch_size=32, num_workers=4)

    print(len(train_data))

    for i, (imgs, _) in enumerate(train_data):
        print(imgs.shape)
        break

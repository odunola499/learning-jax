import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data[idx]
        image = np.array(row['image']).astype(np.float32) / 255
        label = np.array(row['label'])
        return {
            'image': image,
            'label': label
        }

def collate_fn(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    images = np.stack(images)
    images = images[...,None]
    labels = np.stack(labels)
    return {'image': images, 'label': labels}

def get_loaders(batch = 2):
    train_dataset = load_dataset('ylecun/mnist', split='train')
    valid_dataset = load_dataset('ylecun/mnist', split = 'test')
    train_dataset = Data(train_dataset)
    valid_dataset = Data(valid_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, valid_dataloader

if __name__ == "__main__":
    train_dataloader, valid_dataloader = get_loaders()
    print(len(train_dataloader), len(valid_dataloader))
    for batch in train_dataloader:
        print(batch['image'].shape, batch['label'].shape)
        break


# main_loader.py

from torch.utils.data import DataLoader
from dataset.image_split_dataset import RamanImageDatasetSplit

# 血浆数据加载
train_ds_plasma = RamanImageDatasetSplit('images_plasma', phase='train', method='rp')
test_ds_plasma = RamanImageDatasetSplit('images_plasma', phase='test', method='rp')

train_loader_plasma = DataLoader(train_ds_plasma, batch_size=16, shuffle=True)
test_loader_plasma = DataLoader(test_ds_plasma, batch_size=16, shuffle=False)

# 卵泡液数据加载
train_ds_follicular = RamanImageDatasetSplit('images_follicular', phase='train', method='rp')
test_ds_follicular = RamanImageDatasetSplit('images_follicular', phase='test', method='rp')

train_loader_follicular = DataLoader(train_ds_follicular, batch_size=16, shuffle=True)
test_loader_follicular = DataLoader(test_ds_follicular, batch_size=16, shuffle=False)

from nerf_datasets import llff_dataset, blender_dataset
import torch


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __getitem__(self, idx, eval=False):
        return self.dataset.__getitem__(self.indices[idx], eval=eval)
    
    def __len__(self):
        return len(self.indices)
    
    def __repr__(self):
        return f"Subset of {self.dataset} with indices in {self.indices}"
    
    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

def build_dataset(cfg, test_spiral=False):
    if cfg['name'] == 'llff':
        dataset = llff_dataset.LLFFDataset(cfg)
        
        if 'train_test_split' in cfg:        
            if cfg['train_test_split']['uniform_interval']:
                print("Sampling test images uniformly.")
                test_size = int(cfg['train_test_split']['test_size'] * len(dataset))
                interval_size = int(len(dataset) / test_size)
                test_indices = list(range(0, len(dataset), interval_size))
                train_indices = list(set(range(len(dataset))) - set(test_indices))
            else:
                print("Sampling test images randomly.")
                test_size = int(cfg['train_test_split']['test_size'] * len(dataset))
                train_size = len(dataset) - test_size
                
                indices = torch.randperm(len(dataset))
                train_indices = indices[:train_size]
                test_indices = indices[train_size:]
                
                train_indices.sort()
                test_indices.sort()
                
            print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)
        else:
            print("No train-test split provided. Using the entire dataset for training.")
            train_dataset = dataset
            test_dataset = None

    elif cfg['name'] == 'blender':
        train_dataset = blender_dataset.BlenderDataset(cfg, 'train')
        # val_dataset = blender_dataset.BlenderDataset(cfg, 'val')
        test_dataset = blender_dataset.BlenderDataset(cfg, 'test')
    else:
        raise NotImplementedError(f"Dataset {cfg['name']} not implemented.")
        
    return train_dataset, test_dataset


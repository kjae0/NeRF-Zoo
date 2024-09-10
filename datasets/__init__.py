from datasets import llff_dataset

def build_dataset(cfg, test_spiral=False):
    if cfg['name'] == 'llff':
        return llff_dataset.LLFFDataset(cfg)
    else:
        raise NotImplementedError(f"Dataset {cfg['name']} not implemented")


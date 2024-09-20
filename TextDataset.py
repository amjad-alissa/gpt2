from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': (self.inputs[idx] != 0).long(),  # Create attention mask
            'labels': self.targets[idx]
        }

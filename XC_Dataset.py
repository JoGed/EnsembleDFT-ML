import torch
class XC_Dataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        data = self.datasets.copy()
        keys = list(data.keys())
        if "test_params" in keys:
            del data['test_params']
            return [{key: data[key][i] for key in data.keys()}, self.datasets["test_params"]]
        else:
            return {key: self.datasets[key][i] for key in self.datasets.keys()}

    def __len__(self):
        return len(self.datasets["E_tot"])

import torch
class XC_DatasetJump(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        max_dim = torch.amax(self.datasets['dim'])
        max_dim_1stidx = torch.where(self.datasets['dim'] == max_dim)[0][0]
        self.dim_arr = self.datasets['dim'][0:max_dim_1stidx + 1]

    def __getitem__(self, i):
        data = self.datasets.copy()
        keys = list(data.keys())

        if "test_params" in keys:
            del data['test_params']
            return [{key: data[key][i] for key in data.keys()}, self.datasets["test_params"]]

        else:
            return {key: self.datasets[key][i*len(self.dim_arr):i*len(self.dim_arr)+len(self.dim_arr)] for key in self.datasets.keys()}

    def __len__(self):
        return int(len(self.datasets["E_tot"]) / len(self.dim_arr))

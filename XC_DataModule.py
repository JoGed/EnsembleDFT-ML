import torch
from XC_Dataset import XC_Dataset
from XC_DatasetJump import XC_DatasetJump
from torch.utils.data import DataLoader
import pickle
import gzip
import pytorch_lightning as pl
import numpy as np
import os
import sys
from tqdm import tqdm

class XC_DataModule(pl.LightningDataModule):

  def __init__(self, data_dir, batch_size, idxs_ExtPots, DimsToTrain,
               CheckIntegError_Str="", num_workers=2, test_params=None, train_jump=False):
    '''
    Parameters
    ----------
    batch_size: int
        batch_size of the trainings set

    idxs_ExtPots: array_like
        List of 3 lists containing start and end index of the arrangement
        of external potentials in the data set for training-, validation-, and test-set respectively.

    DimsToTrain: array_like
        1D array  containing the fractional dimensions for the training

    CheckIntegError_Str: str
        Determines the the speficic density for what the maximum integral error will be printed

    num_workers: int
        number of data loading workers (default: 2)

    Example
    --------
    For a dataset containing 7 fractional densities [1, 1.2,1.5, 2. ,2.2, 2.5 ,3.] per external potential:
    '>>>  XC_DataModule(batch_size=14, idxs_ExtPots = [[0,100], [100,114] , [140,224]],
                        DimsToTrain=[1.5, 2.2, 3.], CheckIntegError_Str="Dens_total", num_workers=2)'
    '''
    super().__init__()
    self.data_dir            = data_dir
    self.train_jump          = train_jump
    self.batch_size          = batch_size
    self.idxs_ExtPots        = idxs_ExtPots
    self.idxs_Fracs          = self.idxs_ExtPots.copy()
    self.DimsToTrain         = np.array(DimsToTrain)
    self.CheckIntegError_Str = CheckIntegError_Str
    self.num_workers         = num_workers
    self.test_params         = test_params
    self.a = 0   
  def IntegError(self, dens_key):
      '''
      :param dens_key: str
          key_str of a certain density in the data file (e.g. "Data_up")
      Returns: Maximum error of the integrals
      '''
      dx = (self.data["points"][0][1] - self.data["points"][0][0])
      IntegsErrors = len(self.data[dens_key]) * [None]

      for i in tqdm(range(len(IntegsErrors))):
          integs_exact = np.concatenate((self.dim_arr[0:int(len(self.dim_arr) / 2)] - 1, self.dim_arr))
          s = None
          Dens_integ = self.data[dens_key][i].sum().item() * dx
          Dims_Diff = np.abs(np.array(len(integs_exact) * [Dens_integ]) - integs_exact)
          if len(Dims_Diff[np.where(Dims_Diff < 1e-5)]) > 0:
              s = np.array(integs_exact)[np.where(Dims_Diff < 1e-5)][0]
          IntegsErrors[i] = np.abs(Dens_integ - s)
          #print("\n", Dens_integ.item(), "-->", s)
          #----------New Normalization----------------------
          if Dens_integ == 0: self.data[dens_key][i] = 0 * self.data[dens_key][i]
          else: self.data[dens_key][i] = self.data[dens_key][i] / Dens_integ
          self.data[dens_key][i] = s * self.data[dens_key][i]
          #-------------------------------------------------
      return np.amax(IntegsErrors)

  def prepare_data(self):
      data_file = gzip.open(os.getcwd() + self.data_dir, 'rb')
      self.data = pickle.load(data_file)
      data_file.close()
      max_dim = torch.amax(self.data['dim'])
      max_dim_1stidx = torch.where(self.data['dim'] == max_dim)[0][0]
      self.dim_arr = np.array(self.data['dim'])[0:max_dim_1stidx + 1]

      #----------New Normalization----------------------
      print("Renormalizing...")
      self.IntegError("Dens_total")
      self.IntegError("Dens_up")
      self.IntegError("Dens_down")
      #-------------------------------------------------
      #self.CheckIntegError_Str = "Dens_total"
      if len(self.CheckIntegError_Str) > 0: # check integ error
          # keys: "Dens_total", "Dens_up", "Dens_down"
          print("Check integral error for "+ self.CheckIntegError_Str +"...")
          print("Max. integral error for " + self.CheckIntegError_Str + ":", \
                 self.IntegError(dens_key=self.CheckIntegError_Str))

  def setup(self, stage=None):
    if stage == 'fit' or stage is None:
        for j in range(2):
            self.idxs_Fracs[j] = np.array([np.array([np.where(self.dim_arr == dims)[0][0]
                                                     for dims in self.DimsToTrain]) + n * len(self.dim_arr)
                                           for n in range(self.idxs_Fracs[j][0], self.idxs_Fracs[j][-1])]).flatten()
        self.train_m = {key: self.data.copy()[key][self.idxs_Fracs[0]]
                                   for key in list(self.data.copy().keys())}
        self.val_m   = {key: self.data.copy()[key][self.idxs_Fracs[1]]
                                   for key in list(self.data.copy().keys())}
        if self.train_jump:
            self.train_m = XC_DatasetJump(self.train_m)
            self.val_m   = XC_DatasetJump(self.val_m)
        else:
            self.train_m = XC_Dataset(self.train_m)
            self.val_m   = XC_Dataset(self.val_m)

        #print(self.idxs_Fracs[0])
        #print(self.idxs_Fracs[1])
        #sys.exit()

    if stage == 'test' or stage is None:
        self.idxs_Fracs[2] = np.array([np.array([np.where(self.dim_arr == dims)[0][0]  # test_set -> [1,2,3] fractionals
                                                 for dims in [1, 2, 3]]) + n * len(self.dim_arr)
                                       for n in range(self.idxs_Fracs[2][0], self.idxs_Fracs[2][-1])]).flatten()
        test_dic = {}
        test_dic.update({key: self.data.copy()[key][self.idxs_Fracs[2]] for key in list(self.data.copy().keys())})
        test_dic.update({"test_params": self.test_params})

        self.test_m = XC_Dataset(test_dic)
        #print(self.idxs_Fracs[2])
        #sys.exit()

  def train_dataloader(self):
      return DataLoader(self.train_m, batch_size=self.batch_size if not self.train_jump else 1,
                              shuffle=True, num_workers=self.num_workers, drop_last=True)

  def val_dataloader(self):
      return DataLoader(self.val_m, batch_size=self.batch_size if not self.train_jump else 1, shuffle=False,
                              num_workers=self.num_workers, drop_last=True)

  def test_dataloader(self):
    return DataLoader(self.test_m,  batch_size=3, shuffle=False, num_workers=self.num_workers, drop_last=False)

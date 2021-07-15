import os
import torch
from torch import nn
import pytorch_lightning as pl
from argparse import ArgumentParser
import KohnShamSpin
from tqdm import tqdm
from scipy import optimize
import sys
import numpy as np

class XC_Model(pl.LightningModule):

  def __init__(self, **hparams):

    """
    Pass in parsed HyperOptArgumentParser to the model
    :param hparams:
    """
    super().__init__()
    self.save_hyperparameters()
    # -------------------------------
    #  Define Layer Architecture
    # -------------------------------
    self.channels = 1 if not self.hparams.Spin else 2
    if self.hparams.Disc: self.channels += 1
    self.input_dim_Slope  = 1 if not self.hparams.Spin else 2
    self.Conv_Channels    = [self.channels] +  list(self.hparams.Conv_OutChannels)
    self.Slope_Channels   = [self.input_dim_Slope] + list(self.hparams.Conv_OutChannels)
    self.LDA_LayerDims    = [self.hparams.LDA_in_channels] + list(self.hparams.LDA_LayerOutDims)
    self.padding          = int((self.hparams.kernelsize - 1)/2)

    self.Con0 = nn.ModuleList([nn.Conv1d(in_channels=self.Conv_Channels[i], out_channels=self.Conv_Channels[i+1],
                                         kernel_size=self.hparams.kernelsize, padding=self.padding)
                                   for i in range(len(self.Conv_Channels)-1)])
    self.Slope = nn.ModuleList([nn.Conv1d(in_channels=self.Slope_Channels[i], out_channels=self.Slope_Channels[i+1],
                                         kernel_size=self.hparams.kernelsize, padding=self.padding)
                                   for i in range(len(self.Slope_Channels)-1)])

    self.convPreLDA2 = nn.Conv1d(in_channels=self.channels, out_channels=self.hparams.LDA_in_channels,
                                 kernel_size=self.hparams.kernelsize, padding=self.padding, bias=True)
    self.convPreLDA1 = nn.Conv1d(in_channels=1 if not self.hparams.Spin else 2, out_channels=self.hparams.LDA_in_channels,
                                 kernel_size=self.hparams.kernelsize, padding=self.padding, bias=True)

    self.LDA1 = nn.ModuleList([nn.Linear(self.LDA_LayerDims[i], self.LDA_LayerDims[i+1])
                               for i in range(len(self.LDA_LayerDims)-1)])
    self.LDA2 = nn.ModuleList([nn.Linear(self.LDA_LayerDims[i], self.LDA_LayerDims[i+1])
                               for i in range(len(self.LDA_LayerDims)-1)])

    self.actsConvs = nn.ModuleList([nn.SiLU() for _ in range(len(self.Conv_Channels))])
    self.actsLDA   = nn.ModuleList([nn.SiLU() for _ in range(len(self.LDA_LayerDims))])

  def DisconFun(self, pointsAndDens, eps):
      # ---------------------------------------
      #  non-differentiable auxiliary function
      # ---------------------------------------
      points, x = pointsAndDens
      dx = points[0][1] - points[0][0]
      x_slope = x
      for fc, act in zip(self.Slope, self.actsConvs):
          x_slope= act(fc(x_slope))
      return 1 + ((torch.sum(x_slope, (1, 2))) *
                   torch.abs(torch.sin((x.sum(dim=(1, 2)) * dx - eps) * np.pi)) /
                  (x.sum(dim=(1, 2)) * dx + 2)
                  ).unsqueeze(-1)

  def forward(self, pointsAndDens):

      points, x = pointsAndDens
      dx = points[0][1] - points[0][0]

      if self.hparams.Disc:
          if self.hparams.WindowPlusLDA:
              u = self.convPreLDA1(x)
              u = torch.transpose(u, 1, 2)  # -> [B, dim_out , channels]
              for fc, act in zip(self.LDA1, self.actsLDA):
                  u = act(fc(u))
              u = torch.transpose(u, 1, 2)
              u = 1 + ((torch.sum(u, (1, 2))) *
                   torch.abs(torch.sin((x.sum(dim=(1, 2)) * dx - 1e-14) * np.pi)) /
                  (x.sum(dim=(1, 2)) * dx + 2)
                  ).unsqueeze(-1)
              f = self.convPreLDA2(torch.cat((x, u.repeat(1, points.shape[-1]).unsqueeze(1)), dim=1))
              f = torch.transpose(f, 1, 2)
              for fc, act in zip(self.LDA2, self.actsLDA):
                  f = act(fc(f))
              f = torch.transpose(f, 1, 2)
              x_out = -0.5 * torch.sum(torch.multiply(x.sum(1).unsqueeze(1), f), -1)  # -> [B, 1]


          else:
              u = self.DisconFun((points, x), eps=1e-14).repeat(1, points.shape[-1]).unsqueeze(1)
              f = torch.cat((x, u), dim=1)
              for fc, act in zip(self.Con0,  self.actsConvs):
                  f = act(fc(f))
              x_out = -0.5 * torch.sum(torch.multiply(x.sum(1).unsqueeze(1), f), -1)

      else:
          if self.hparams.WindowPlusLDA:
              f = self.convPreLDA2(x)
              f = torch.transpose(f, 1, 2)
              for fc, act in zip(self.LDA2, self.actsLDA):
                  f = act(fc(f))
              f = torch.transpose(f, 1, 2)
              x_out = -0.5 * torch.sum(torch.multiply(x.sum(1).unsqueeze(1), f), -1)

          else:
              f = x
              for fc, act in zip(self.Con0, self.actsConvs):
                  f = act(fc(f))
              x_out = -0.5 * torch.sum(torch.multiply(x.sum(1).unsqueeze(1), f), -1)

      return x_out

  def loss(self, E_xc_out, E_xc_ref, V_xc_out, V_xc_ref):
      E_xc_ref = E_xc_ref.unsqueeze(1)
      MSE1 = nn.MSELoss(reduction="mean")
      l_E_xc = MSE1(E_xc_out, E_xc_ref)
      l_V_xc = MSE1(V_xc_out, V_xc_ref)

      return l_E_xc, l_V_xc, 10*l_E_xc +100*l_V_xc

  def lossJump(self, Dens_total_1,    Dens_total_2,    V_xc_out1,     V_xc_out2,
                     Dens_total_mix1, Dens_total_mix2, V_xc_out_mix1, V_xc_out_mix2,
                     evals_Int1, evals_Int2, E_tot_Triplet):

      IONminusAFF1   = E_tot_Triplet[0][1]   - 2 * E_tot_Triplet[0][0] + 0
      IONminusAFF2   = E_tot_Triplet[0][2]   - 2 * E_tot_Triplet[0][1] + E_tot_Triplet[0][0]
      KSgap1         = evals_Int1[0][0] - evals_Int1[0][0] #?????
      KSgap2         = evals_Int2[0][1] - evals_Int2[0][0]
      JumpXC1        = IONminusAFF1 - KSgap1
      JumpXC2        = IONminusAFF2 - KSgap2
      #print(JumpXC1.item(), JumpXC2.item())
      MSE    = nn.MSELoss(reduction="mean")
      Id_fun = torch.ones(V_xc_out1.shape[-1]).view(1,1,-1)
      if self.hparams.gpus_num != 0 or len(self.hparams.gpus_devices) != 0:
          Id_fun = Id_fun.cuda()
      JumpXC1_fun = JumpXC1 * Id_fun
      JumpXC2_fun = JumpXC2 * Id_fun
      JumpLoss1   = MSE((V_xc_out_mix1 - V_xc_out1), JumpXC1_fun)
      JumpLoss2   = MSE((V_xc_out_mix2 - V_xc_out2), JumpXC2_fun)
      return JumpLoss1, JumpLoss2

  def training_step(self, batch, batch_idx):
    eps=1.1e-14
    if hasattr(self.hparams, "train_jump"):
        if self.hparams.train_jump:
            batch = {key: batch[key].squeeze(0) for key in batch.keys()}
    points        = batch["points"]
    Dens_total    = batch["Dens_total"]
    Dens_up       = batch["Dens_up"]
    Dens_down     = batch["Dens_down"]
    V_ext         = batch["v_ext"]
    V_xc_NoSpin   = batch["v_xc_NoSpin"]
    V_xc_up       = batch["v_xc_up"]
    V_xc_down     = batch["v_xc_down"]
    E_xc_NoSpin   = batch["E_xc_NoSpin"]
    E_xc_Spin     = batch["E_xc_Spin"]
    E_tot         = batch["E_tot"]
    evals_Int1    = batch["evals_Int1"]
    evals_Int2    = batch["evals_Int2"]
    E_tot_Triplet = batch["E_tot_Triplet"]
    dx = points[0][1] - points[0][0]
    # ------------------------------------------------------------------------------------------------------------------
    #  Compute E_xc & V_xc
    # ------------------------------------------------------------------------------------------------------------------
    if self.hparams.Spin:
        DensUpAndDown = torch.cat((Dens_up.unsqueeze(1), Dens_down.unsqueeze(1)), 1)
        V_xcUpAndDown = torch.cat((V_xc_up.unsqueeze(1), V_xc_down.unsqueeze(1)), 1)
        DensUpAndDown.requires_grad = True
        E_xc_Spin_out = self((points, DensUpAndDown))
        E_xc_Spin_out_deriv = \
            torch.autograd.grad(inputs=DensUpAndDown, outputs=E_xc_Spin_out, create_graph=True,
                                retain_graph=True, grad_outputs=torch.ones_like(E_xc_Spin_out))[0] / dx
        l_Exc, l_V_xc, loss = self.loss(E_xc_out=E_xc_Spin_out, E_xc_ref=E_xc_Spin, V_xc_out=E_xc_Spin_out_deriv,
                                        V_xc_ref=V_xcUpAndDown)

        if hasattr(self.hparams, "SpinMirror"):
            if self.hparams.SpinMirror:
                DensUpAndDown_mirr = torch.cat((Dens_down.detach().clone().unsqueeze(1), Dens_up.detach().clone().unsqueeze(1)), 1)
                V_xcUpAndDown_mirr  = torch.cat((V_xc_down.detach().clone().unsqueeze(1), V_xc_up.detach().clone().unsqueeze(1)), 1)
                DensUpAndDown_mirr.requires_grad = True
                E_xc_Spin_out_mirr = self((points, DensUpAndDown_mirr))
                E_xc_Spin_out_deriv_mirr  = \
                    torch.autograd.grad(inputs=DensUpAndDown_mirr , outputs=E_xc_Spin_out_mirr , create_graph=True,
                                        retain_graph=True, grad_outputs=torch.ones_like(E_xc_Spin_out_mirr))[0] / dx
                l_Exc_mirr , l_V_xc_mirr , loss_mirr  = self.loss(E_xc_out=E_xc_Spin_out_mirr, E_xc_ref=E_xc_Spin,
                                                V_xc_out=E_xc_Spin_out_deriv_mirr,
                                                V_xc_ref=V_xcUpAndDown_mirr)

                l_Exc  = (l_Exc  + l_Exc_mirr)  / 2.
                l_V_xc = (l_V_xc + l_V_xc_mirr) / 2.
                loss   = (loss   + loss_mirr)   / 2.


    else:
        Dens_total  = Dens_total.unsqueeze(1)
        V_xc_NoSpin = V_xc_NoSpin.unsqueeze(1)
        Dens_total.requires_grad = True
        E_xc_NoSpin_out = self((points, Dens_total))
        E_xc_NoSpin_out_deriv = \
            torch.autograd.grad(inputs=Dens_total, outputs=E_xc_NoSpin_out, create_graph=True,
                                retain_graph=True, grad_outputs=torch.ones_like(E_xc_NoSpin_out))[0] / dx

        l_Exc, l_V_xc, loss = self.loss(E_xc_out=E_xc_NoSpin_out, E_xc_ref=E_xc_NoSpin, V_xc_out=E_xc_NoSpin_out_deriv,
                                        V_xc_ref=V_xc_NoSpin)

        if hasattr(self.hparams, "train_jump"):
            if self.hparams.train_jump:
                mid_idx = int(len(self.hparams.DimsToTrain) / 2)
                Dens_total_mix1 = (1 - eps) * Dens_total.detach().clone()[0] + eps * Dens_total.detach().clone()[mid_idx]
                Dens_total_mix2 = (1 - eps) * Dens_total.detach().clone()[mid_idx] + eps * Dens_total.detach().clone()[-1]
                # ---------------------------------------------------------
                Dens_total_mix1 = Dens_total_mix1.unsqueeze(0)
                Dens_total_mix1.requires_grad = True
                E_xc_NoSpin_out_mix1 = self((points, Dens_total_mix1))
                E_xc_NoSpin_out_deriv_mix1 = \
                    torch.autograd.grad(inputs=Dens_total_mix1, outputs=E_xc_NoSpin_out_mix1, create_graph=True,
                                        retain_graph=True, grad_outputs=torch.ones_like(E_xc_NoSpin_out_mix1))[0] / dx
                # ---------------------------------------------------------
                Dens_total_mix2 = Dens_total_mix2.unsqueeze(0)
                Dens_total_mix2.requires_grad = True
                E_xc_NoSpin_out_mix2 = self((points, Dens_total_mix2))
                E_xc_NoSpin_out_deriv_mix2 = \
                    torch.autograd.grad(inputs=Dens_total_mix2, outputs=E_xc_NoSpin_out_mix2, create_graph=True,
                                        retain_graph=True, grad_outputs=torch.ones_like(E_xc_NoSpin_out_mix2))[0] / dx
                # ---------------------------------------------------------
                lossJump1, lossJump2 = self.lossJump(Dens_total_1=Dens_total[0].unsqueeze(0),
                                                     Dens_total_2=Dens_total[mid_idx].unsqueeze(0),
                                                     V_xc_out1=E_xc_NoSpin_out_deriv[0].unsqueeze(0),
                                                     V_xc_out2=E_xc_NoSpin_out_deriv[mid_idx].unsqueeze(0),
                                                     Dens_total_mix1=Dens_total_mix1,
                                                     Dens_total_mix2=Dens_total_mix2,
                                                     V_xc_out_mix1=E_xc_NoSpin_out_deriv_mix1,
                                                     V_xc_out_mix2=E_xc_NoSpin_out_deriv_mix2,
                                                     evals_Int1=evals_Int1,
                                                     evals_Int2=evals_Int2,
                                                     E_tot_Triplet=E_tot_Triplet)
                self.log('val Jump1', lossJump1)
                self.log('val Jump2', lossJump2)
                loss += 10 * (lossJump1 + lossJump2)

    self.log('train_loss',    loss, prog_bar=True)
    self.log('train MSE EXC', l_Exc)
    self.log('train MSE VXC', l_V_xc)

    # ------------------------------------------------------------------------------------------------------------------
    return loss

  def validation_step(self, batch, batch_idx):
      eps = 1.1e-14

      if hasattr(self.hparams, "train_jump"):
          if self.hparams.train_jump:
              batch = {key: batch[key].squeeze(0) for key in batch.keys()}

      points        = batch["points"]
      Dens_total    = batch["Dens_total"]
      Dens_up       = batch["Dens_up"]
      Dens_down     = batch["Dens_down"]
      V_ext         = batch["v_ext"]
      V_xc_NoSpin   = batch["v_xc_NoSpin"]
      V_xc_up       = batch["v_xc_up"]
      V_xc_down     = batch["v_xc_down"]
      E_xc_NoSpin   = batch["E_xc_NoSpin"]
      E_xc_Spin     = batch["E_xc_Spin"]
      E_tot         = batch["E_tot"]
      evals_Int1    = batch["evals_Int1"]
      evals_Int2    = batch["evals_Int2"]
      E_tot_Triplet = batch["E_tot_Triplet"]
      
      dx = points[0][1] - points[0][0]

      torch.set_grad_enabled(True)

      # ----------------------------------------------------------------------------------------------------------------
      #  Compute E_xc & V_xc
      # ----------------------------------------------------------------------------------------------------------------
      if self.hparams.Spin:
          DensUpAndDown = torch.cat((Dens_up.unsqueeze(1), Dens_down.unsqueeze(1)), 1)
          V_xcUpAndDown = torch.cat((V_xc_up.unsqueeze(1), V_xc_down.unsqueeze(1)), 1)
          DensUpAndDown.requires_grad = True
          E_xc_Spin_out = self((points, DensUpAndDown))
          E_xc_Spin_out_deriv = \
              torch.autograd.grad(inputs=DensUpAndDown, outputs=E_xc_Spin_out, create_graph=True,
                                  retain_graph=True, grad_outputs=torch.ones_like(E_xc_Spin_out))[0] / dx

          l_Exc, l_V_xc, loss = self.loss(E_xc_out=E_xc_Spin_out, E_xc_ref=E_xc_Spin, V_xc_out=E_xc_Spin_out_deriv,
                                          V_xc_ref=V_xcUpAndDown)

          self.log('val_loss', loss, prog_bar=True)
          self.log('val MSE EXC', l_Exc)
          self.log('val MSE VXC', l_V_xc)

          return {"val_loss": loss, "Tuple": (points, DensUpAndDown, V_ext, E_xc_Spin_out,
                                                  E_xc_Spin_out_deriv, V_xcUpAndDown, E_tot, E_xc_Spin, loss)}
      else:
          Dens_total  = Dens_total.unsqueeze(1)
          V_xc_NoSpin = V_xc_NoSpin.unsqueeze(1)
          Dens_total.requires_grad = True

          E_xc_NoSpin_out = self((points, Dens_total))
          E_xc_NoSpin_out_deriv = \
              torch.autograd.grad(inputs=Dens_total, outputs=E_xc_NoSpin_out, create_graph=False,
                                  retain_graph=True, grad_outputs=torch.ones_like(E_xc_NoSpin_out))[0] / dx

          l_Exc, l_V_xc, loss = self.loss(E_xc_out=E_xc_NoSpin_out, E_xc_ref=E_xc_NoSpin, V_xc_out=E_xc_NoSpin_out_deriv,
                                          V_xc_ref=V_xc_NoSpin)
          self.log('val MSE EXC', l_Exc)
          self.log('val MSE VXC', l_V_xc)

          if hasattr(self.hparams, "train_jump"):
              if self.hparams.train_jump:
                  mid_idx = int(len(self.hparams.DimsToTrain) / 2)
                  Dens_total_mix1 = (1 - eps) * Dens_total.detach().clone()[0] + eps * Dens_total.detach().clone()[mid_idx]
                  Dens_total_mix2 = (1 - eps) * Dens_total.detach().clone()[mid_idx] + eps * Dens_total.detach().clone()[-1]
                  Dens_total_mix1 = Dens_total_mix1.unsqueeze(0)
                  Dens_total_mix1.requires_grad = True
                  E_xc_NoSpin_out_mix1 = self((points, Dens_total_mix1))
                  E_xc_NoSpin_out_deriv_mix1 = \
                      torch.autograd.grad(inputs=Dens_total_mix1, outputs=E_xc_NoSpin_out_mix1, create_graph=True,
                                          retain_graph=True, grad_outputs=torch.ones_like(E_xc_NoSpin_out_mix1))[0] / dx
                  # ---------------------------------------------------------
                  Dens_total_mix2 = Dens_total_mix2.unsqueeze(0)
                  Dens_total_mix2.requires_grad = True
                  E_xc_NoSpin_out_mix2 = self((points, Dens_total_mix2))
                  E_xc_NoSpin_out_deriv_mix2 = \
                      torch.autograd.grad(inputs=Dens_total_mix2, outputs=E_xc_NoSpin_out_mix2, create_graph=True,
                                          retain_graph=True, grad_outputs=torch.ones_like(E_xc_NoSpin_out_mix2))[0] / dx
                  # ---------------------------------------------------------
                  lossJump1, lossJump2 = self.lossJump(Dens_total_1=Dens_total[0].unsqueeze(0),
                                                       Dens_total_2=Dens_total[mid_idx].unsqueeze(0),
                                                       V_xc_out1=E_xc_NoSpin_out_deriv[0].unsqueeze(0),
                                                       V_xc_out2=E_xc_NoSpin_out_deriv[mid_idx].unsqueeze(0),
                                                       Dens_total_mix1=Dens_total_mix1,
                                                       Dens_total_mix2=Dens_total_mix2,
                                                       V_xc_out_mix1=E_xc_NoSpin_out_deriv_mix1,
                                                       V_xc_out_mix2=E_xc_NoSpin_out_deriv_mix2,
                                                       evals_Int1=evals_Int1,
                                                       evals_Int2=evals_Int2,
                                                       E_tot_Triplet=E_tot_Triplet)
                  self.log('val Jump1', lossJump1)
                  self.log('val Jump2', lossJump2)
                  loss += 10 * (lossJump1 + lossJump2)
                  # ---------------------------------------------------------
          self.log('val_loss', loss, prog_bar=True)
          return {"val_loss": loss, "Tuple": (points, Dens_total, V_ext, E_xc_NoSpin_out, E_xc_NoSpin_out_deriv, V_xc_NoSpin,
                                              E_tot, E_xc_NoSpin, loss)}

  def GS_dens_splitter(self, particles):
      # ------------------------------------------------------
      # returns occupation (array_like) of Kohn Sham orbitals
      # ------------------------------------------------------
      if particles < 1 - 1e-14:
          raise Exception("particles < 1!")
      rounded, append = int(particles), particles - int(particles)
      if rounded % 2 == 0:
          s = int(rounded / 2.)
          up_occ = np.ones(s + 1)
          up_occ[-1] = append
          down_occ = np.ones(s + 1)
          down_occ[-1] = 0
      else:
          s = int((rounded - 1) / 2.)
          up_occ = np.ones(s + 1)
          down_occ = np.ones(s + 1)
          down_occ[-1] = append
      return up_occ + down_occ, up_occ, down_occ

  def KSIterations(self, KSsystem, Dens_inp, V_xc_inp, E_xc):
      # ---------------------------------------------------------------------------------------------------------
      # SOLVING KOHN SHAM EQUATIONS
      # ---------------------------------------------------------------------------------------------------------
      Dens_KS_init = Dens_inp.detach().cpu().clone().numpy()
      V_xc_in = V_xc_inp.detach().cpu().clone().numpy()
      KSsystem["v_ext"] = KSsystem["v_ext"].cpu().numpy()
      KSsystem['points'] = KSsystem['points'].cpu().numpy()
      v_ext_diag = KSsystem['v_ext']

      def selfcons(x):
          x_last = x.copy()
          if self.hparams.Spin:
              v_H_diag = KohnShamSpin.V_Hartree(KSsystem, x[0] + x[1])
              v_eff_diag_up = v_H_diag + v_ext_diag + V_xc_in[0]
              v_eff_diag_down = v_H_diag + v_ext_diag + V_xc_in[1]
              evals_up, selfcons.Psi_up, D_Matrix_up = KohnShamSpin.Orbitals(P=KSsystem,
                                                                             v_eff_diag=v_eff_diag_up,
                                                                             occ=KSsystem["up_occ"])
              evals_down, selfcons.Psi_down, D_Matrix_down = KohnShamSpin.Orbitals(P=KSsystem,
                                                                                   v_eff_diag=v_eff_diag_down,
                                                                                   occ=KSsystem["down_occ"])

              Dens_KS_new_up = KohnShamSpin.Density(P=KSsystem, D_arr=D_Matrix_up, occ=KSsystem["up_occ"])
              Dens_KS_new_down = KohnShamSpin.Density(P=KSsystem, D_arr=D_Matrix_down, occ=KSsystem["down_occ"])
              Dens_KS_new = np.stack((Dens_KS_new_up, Dens_KS_new_down))

          else:
              v_H_diag = KohnShamSpin.V_Hartree(KSsystem, x[0])
              v_eff_diag = v_H_diag + v_ext_diag + V_xc_in[0]
              evals, selfcons.Psi, D_Matrix = KohnShamSpin.Orbitals(P=KSsystem,
                                                                    v_eff_diag=v_eff_diag,
                                                                    occ=KSsystem["occ"])
              Dens_KS_new = KohnShamSpin.Density(P=KSsystem, D_arr=D_Matrix, occ=KSsystem["occ"])
          return Dens_KS_new - x_last

      Dens_KS_out = torch.tensor(optimize.broyden1(selfcons, Dens_KS_init, f_tol=1e-8))
      E_ext_KS, E_H_KS = KohnShamSpin.Energies(P=KSsystem,
                                               Dens=torch.sum(Dens_KS_out, dim=0),
                                               v_ext=KSsystem["v_ext"],
                                               v_H=KohnShamSpin.V_Hartree(KSsystem, torch.sum(Dens_KS_out, dim=0)))

      if not self.hparams.Spin:
          E_kin_KS = KohnShamSpin.E_kinetic(P=KSsystem, Psi=selfcons.Psi, occ=KSsystem["occ"])
          E_tot_KS = E_xc + E_kin_KS + E_ext_KS + E_H_KS

      else:
          E_kin_KS_up = KohnShamSpin.E_kinetic(P=KSsystem, Psi=selfcons.Psi_up, occ=KSsystem["up_occ"])
          E_kin_KS_down = KohnShamSpin.E_kinetic(P=KSsystem, Psi=selfcons.Psi_down, occ=KSsystem["down_occ"])
          E_tot_KS = E_xc + (E_kin_KS_up + E_kin_KS_down) + E_ext_KS + E_H_KS

      return Dens_KS_out, E_tot_KS

  def test_step(self, batch, batch_idx):
      import pprint
      import gzip
      import pickle
      import re
      batch, test_params = batch
      test_params = {key: test_params[key][0] for key in test_params.keys()}
      test_params["CompareWith"]=list(test_params["CompareWith"].replace('[', '').
                                       replace(']', '').replace(',', ' ').split())

      if batch_idx == 0:
          print("\n\nTEST PARAMETERS:")
          pprint.pprint(test_params)
          print("\n")
      # ---------------------------------------------------------------------------------------------------------
      # COMPUTE TOTAL ENERGY
      # ---------------------------------------------------------------------------------------------------------
      def CalcXCData(batch):
          eps = 1.1e-14
          points        = batch["points"]
          Dens_total    = batch["Dens_total"]
          Dens_up       = batch["Dens_up"]
          Dens_down     = batch["Dens_down"]
          V_ext         = batch["v_ext"]
          V_xc_NoSpin   = batch["v_xc_NoSpin"]
          V_xc_up       = batch["v_xc_up"]
          V_xc_down     = batch["v_xc_down"]
          E_xc_NoSpin   = batch["E_xc_NoSpin"]
          E_xc_Spin     = batch["E_xc_Spin"]
          E_tot         = batch["E_tot"]
          #print(batch["dim"])

          dx = points[0][1] - points[0][0]
          torch.set_grad_enabled(True)
          # ----------------------------------------------------------------------------------------------------------------
          #  Compute E_xc & V_xc
          # ----------------------------------------------------------------------------------------------------------------
          if self.hparams.Spin:
              DensUpAndDown = torch.cat((Dens_up.unsqueeze(1), Dens_down.unsqueeze(1)), 1)
              V_xcUpAndDown = torch.cat((V_xc_up.unsqueeze(1), V_xc_down.unsqueeze(1)), 1)
              DensUpAndDown.requires_grad = True
              E_xc_Spin_out = self((points, DensUpAndDown))
              E_xc_Spin_out_deriv = \
                  torch.autograd.grad(inputs=DensUpAndDown, outputs=E_xc_Spin_out, create_graph=True,
                                      retain_graph=True, grad_outputs=torch.ones_like(E_xc_Spin_out))[0] / dx

              return points, DensUpAndDown, V_ext, E_xc_Spin_out, E_xc_Spin_out_deriv, V_xcUpAndDown, E_tot, E_xc_Spin

          else:
              Dens_total  = Dens_total.unsqueeze(1)
              V_xc_NoSpin = V_xc_NoSpin.unsqueeze(1)
              Dens_total.requires_grad = True
              E_xc_NoSpin_out = self((points, Dens_total))
              E_xc_NoSpin_out_deriv = \
                  torch.autograd.grad(inputs=Dens_total, outputs=E_xc_NoSpin_out, create_graph=False,
                                      retain_graph=True, grad_outputs=torch.ones_like(E_xc_NoSpin_out))[0] / dx

              return points, Dens_total, V_ext, E_xc_NoSpin_out, E_xc_NoSpin_out_deriv, V_xc_NoSpin, E_tot, E_xc_NoSpin


      points, Dens, V_ext, E_xc_out, E_xc_out_deriv, V_xc, E_tot_ref, E_xc_ref = CalcXCData(batch)
      alpha = torch.linspace(0, 1, test_params["FracPoints"], dtype=torch.float64)
      points     = points[0]
      dx         = points[1] - points[0]

      Exc_mix_arr  = torch.zeros((2, len(alpha)))
      Etot_mix_arr = torch.zeros((2, len(alpha)))

      if self.hparams.Spin:
          D_mix_up_arr      = torch.zeros((2, len(alpha), len(points)))
          D_mix_down_arr    = torch.zeros((2, len(alpha), len(points)))
          Vxc_up_mix_arr    = torch.zeros((2, len(alpha), len(points)))
          Vxc_down_mix_arr  = torch.zeros((2, len(alpha), len(points)))
          D_KS_mix_up_arr   = torch.zeros((2, len(alpha), len(points)))
          D_KS_mix_down_arr = torch.zeros((2, len(alpha), len(points)))

      else:
          D_mix_arr    = torch.zeros((2, len(alpha), len(points)))
          Vxc_mix_arr  = torch.zeros((2, len(alpha), len(points)))
          D_KS_mix_arr = torch.zeros((2, len(alpha), len(points)))

      #mix_doubles = [0, int(len(self.hparams.DimsToTrain) / 2.)]
      mix_doubles = [0, 1]

      for m in range(len(mix_doubles)):
          for i in tqdm(range(len(alpha))):
              if i == 0:  # "left" integer particle number
                  Dens_mix, E_xc_mix, V_xc_mix = Dens[mix_doubles[m]], E_xc_out[mix_doubles[m]], E_xc_out_deriv[
                      mix_doubles[m]]
              elif i == len(alpha) - 1:  # "right" integer particle number
                  Dens_mix, E_xc_mix, V_xc_mix = Dens[mix_doubles[m] + (mix_doubles[1] - mix_doubles[0])], \
                                                 E_xc_out[mix_doubles[m] + (mix_doubles[1] - mix_doubles[0])], \
                                                 E_xc_out_deriv[mix_doubles[m] + (mix_doubles[1] - mix_doubles[0])]
              else:  # fractional particle numbers
                  Dens_mix = (1 - alpha[i]) * Dens[mix_doubles[m]].detach().clone() \
                             + alpha[i] * Dens[mix_doubles[m] + (mix_doubles[1] - mix_doubles[0])].detach().clone()

                  Dens_mix = Dens_mix.unsqueeze(0)
                  Dens_mix.requires_grad = True
                  E_xc_mix = self((points.unsqueeze(0), Dens_mix))
                  V_xc_mix = \
                      torch.autograd.grad(inputs=Dens_mix, outputs=E_xc_mix, create_graph=False,
                                          retain_graph=True, grad_outputs=torch.ones_like(E_xc_mix))[0] / dx

                  Dens_mix = Dens_mix.squeeze(0)
                  E_xc_mix = E_xc_mix.squeeze(0)
                  V_xc_mix = V_xc_mix.squeeze(0)

              # --------------------------------------------------------------------------------------------------------
              # COMPUTE TOTAL ENERGY BY SOLVING KOHN SHAM EQUATIONS
              # --------------------------------------------------------------------------------------------------------
              Dens_integ = Dens_mix.sum().item() * dx

              KSsystem = {
                  "dim":       Dens_integ,
                  "v_ext":     V_ext[mix_doubles[m]],
                  "points":    points,
                  "N":         len(points),
                  "dx":        dx.item(),
                  'laplOrder': 4,
                  "occ":       None,
                  "up_occ":    None,
                  "down_occ":  None,
              }
              # ---------------------------------------------------------------------------------------------------------
              # ENSURE THAT DENSITIES WILL BE ASSIGNED TO CORRECT FRACTIONAL OCCUPATION
              # ---------------------------------------------------------------------------------------------------------
              Dims_Diff = np.abs(np.array(len([1,2,3]) * [KSsystem["dim"]]) - np.array([1,2,3]))
              KSsystem['occ'], KSsystem['up_occ'], KSsystem['down_occ'] = self.GS_dens_splitter(KSsystem['dim'])
              Dens_KS_out, E_tot_mix = self.KSIterations(KSsystem=KSsystem, Dens_inp=Dens_mix, V_xc_inp=V_xc_mix, E_xc=E_xc_mix) 

              Exc_mix_arr[m, i]  = E_xc_mix
              Etot_mix_arr[m, i] = E_tot_mix

              if self.hparams.Spin:
                  D_mix_up_arr[m, i]      = Dens_mix[0]
                  D_mix_down_arr[m, i]    = Dens_mix[1]
                  Vxc_up_mix_arr[m, i]    = V_xc_mix[0]
                  Vxc_down_mix_arr[m, i]  = V_xc_mix[1]
                  D_KS_mix_up_arr[m, i]   = Dens_KS_out[0]
                  D_KS_mix_down_arr[m, i] = Dens_KS_out[1]
              else:
                  D_mix_arr[m, i]    = Dens_mix[0]
                  Vxc_mix_arr[m, i]  = V_xc_mix[0]
                  D_KS_mix_arr[m, i] = Dens_KS_out[0]

      E_t_r  = E_tot_ref.detach().cpu().numpy()
      E_xc_r = E_xc_ref.detach().cpu().numpy()

      def E_tot_exact_func(N):
          #mix_doubles = [0, int(len(self.hparams.DimsToTrain) / 2.)]
          mix_doubles = [0, 1]
          m = 1 if N[-1] > 2.001 else 0
          slope = E_t_r[mix_doubles[m] + (mix_doubles[1] - mix_doubles[0])] - E_t_r[mix_doubles[m]]
          offset = (E_t_r[mix_doubles[m] + (mix_doubles[1] - mix_doubles[0])] + E_t_r[mix_doubles[m]] - slope * (
                  N[0] + N[-1])) * 0.5
          return slope * N + offset

      N_12arr = np.linspace(1, 2, test_params["FracPoints"])
      N_23arr = np.linspace(2, 3, test_params["FracPoints"])
      E_tot_exact12 = E_tot_exact_func(N_12arr)
      E_tot_exact23 = E_tot_exact_func(N_23arr)

      #quadCoeff_12 = np.polyfit(N_12arr, Etot_mix_arr[0].detach().cpu().numpy(), 2)[0]
      #quadCoeff_23 = np.polyfit(N_23arr, Etot_mix_arr[1].detach().cpu().numpy(), 2)[0]
      #L2Normsquared_12 = np.sum((Etot_mix_arr[0].detach().cpu().numpy() - E_tot_exact12) ** 2) * (
      #            alpha[1] - alpha[0]).item()
      #L2Normsquared_23 = np.sum((Etot_mix_arr[1].detach().cpu().numpy() - E_tot_exact23) ** 2) * (
      #            alpha[1] - alpha[0]).item()

      MSE_Etot = nn.MSELoss()(torch.cat((Etot_mix_arr[0].detach(),
                                        Etot_mix_arr[1].detach())),
                              torch.cat((torch.from_numpy(E_tot_exact12),
                                        torch.from_numpy(E_tot_exact23)))).item()
      Var_Etot = torch.var(torch.cat((Etot_mix_arr[0].detach(),
                                      Etot_mix_arr[1].detach()))
                         - torch.cat((torch.from_numpy(E_tot_exact12),
                                      torch.from_numpy(E_tot_exact23)))).item()
      #print(batch_idx, MSE_Etot, Var_Etot)
      # ---------------------------------------------------------------------------------------------------------
      # ERRORS INTEGER ENERGIES
      # ---------------------------------------------------------------------------------------------------------
      E_tot_Triple = torch.tensor(
          (Etot_mix_arr[0, 0], Etot_mix_arr[0, -1], Etot_mix_arr[1, -1]))  # machine output
      E_tot_ref_Triple = E_tot_ref.clone()[[0, mix_doubles[1], -1]]
      E_xc_ref_Triple = E_xc_ref.clone()[[0, mix_doubles[1], -1]]
      E1Diff = (E_tot_Triple[0] - E_tot_ref_Triple[0]).item()
      E2Diff = (E_tot_Triple[1] - E_tot_ref_Triple[1]).item()
      E3Diff = (E_tot_Triple[2] - E_tot_ref_Triple[2]).item()
      Exc1_exact  = E_xc_ref_Triple[0].item()
      Exc2_exact  = E_xc_ref_Triple[1].item()
      Exc3_exact  = E_xc_ref_Triple[2].item()
      Etot1_exact = E_tot_ref_Triple[0].item()
      Etot2_exact = E_tot_ref_Triple[1].item()
      Etot3_exact = E_tot_ref_Triple[2].item()
      Errors1 = E1Diff
      Errors2 = E2Diff
      Errors3 = E3Diff

      Energies_dic = {
          'model_ckpt':     test_params['model_ckpt'],
          "plot_label":     test_params["plot_label"],
          "NPoints":        [N_12arr, N_23arr[1::]],
          "Diffs":          [Etot_mix_arr[0].detach().cpu().numpy() - E_tot_exact12,
                             Etot_mix_arr[1].detach().cpu().numpy()[1::] - E_tot_exact23[1::]],
          "Derivatives":    [np.gradient(Etot_mix_arr[0].detach().cpu().numpy()),
                             np.gradient(Etot_mix_arr[1].detach().cpu().numpy()[1::])],
          "SystemIdx":      test_params["SystemIdx"].item(),
          "ExtPotStartIdx": test_params["idxs_ExtPotsTest"][0].item()
      }
      if test_params["SystemIdx"].item()== batch_idx:
          Energies_NN_file = gzip.open(re.sub('\.ckpt$', '', "".join([os.getcwd(), test_params['model_ckpt']])) +
                                       "_ENERGIES_idx=" + str(Energies_dic["ExtPotStartIdx"])+"_" +
                                       str(test_params["SystemIdx"].item()) +".gz", 'wb')
          pickle.dump(Energies_dic, Energies_NN_file)
          Energies_NN_file.close()


      # ---------------------------------------------------------------------------------------------------------
      # PLOT RESULTS
      # ---------------------------------------------------------------------------------------------------------
      import matplotlib
      from matplotlib import pyplot as plt
      matplotlib.rcParams['mathtext.fontset'] = 'cm'
      matplotlib.rcParams['font.family'] = 'STIXGeneral'
      # ------------------------------------------------
      def floatArray(string):
          return np.array(string.replace('[', '').replace(']', '').replace(',', ' ').split()).astype(float)
      # ------------------------------------------------

      if (test_params["PlotDensitiesDim"] != -1) and (test_params["SystemIdx"].item() == batch_idx):
          N = test_params["PlotDensitiesDim"].item()
          #idx = np.where(self.hparams.DimsToTrain == N)[0][0]

          #print(np.where(np.array([1,2,3]) == N))
          #sys.exit()
          idx = np.where(np.array([1,2,3]) == N)[0][0]
          fig_height = 4
          fig_width = 5
          fvxc, ax = plt.subplots(1, sharex=True, sharey=True)
          fvxc.set_figheight(fig_height)
          fvxc.set_figwidth(fig_width)
          plt.xlabel(r"$x\;[a.u.]$", fontsize=20)
          plt.tick_params(labelsize=15)
          plt.ylabel(r'$\rho, v^{\mathrm{xc}} \;[a.u.]$', fontsize=20)
          plt.tick_params(labelsize=13)
          ax.grid(linestyle='--', linewidth=0.6)
          plt.xlim(-11.5, 11.5)
          s = "up" if self.hparams.Spin else ""
          linew = 2.5
          #print(E_xc_out_deriv[idx][0].detach().cpu(), V_xc[idx][0].detach().cpu().numpy())
          #sys.exit()
          ax.plot(points.detach().cpu(), E_xc_out_deriv[idx][0].detach().cpu(), alpha=0.7, color="r", linewidth=linew,
                  label=r"$v^{\mathrm{xc}}_{\mathrm{" + s + "}, \mathrm{ML}}$" + "$\,(N=$" + str(N) + ")")
          ax.plot(points.detach().cpu(), V_xc[idx][0].detach().cpu(), alpha=1, color="k", linestyle="dashed",
                  linewidth=linew / 2.,
                  label=r"$v^{\mathrm{xc}}_{\mathrm{" + s + "}, \mathrm{Exact}}$" + "$\,(N=$" + str(N) + ")")
          leg1 = ax.legend(loc=(0., 1.02), fontsize=12)

          if self.hparams.Spin:
              s = "down"
              # pl_1, = ax.plot(self.points, Dens[idx][1].detach().cpu().numpy(), alpha=0.5, color="g",
              #        label=r"$\rho_{\mathrm{" + s + "}}$" + "$\,(N=$" + str(N) + ")")
              pl_2, = ax.plot(points.detach().cpu(), E_xc_out_deriv[idx][1].detach().cpu(), alpha=0.7, color="g", linewidth=linew,
                              label=r"$v^{\mathrm{xc}}_{\mathrm{" + s + "}, \mathrm{ML}}$" + "$\,(N=$" + str(N) + ")")
              pl_3, = ax.plot(points.detach().cpu(), V_xc[idx][1].detach().cpu(), alpha=1, color="k", linestyle="dashdot",
                              linewidth=linew / 2.,
                              label=r"$v^{\mathrm{xc}}_{\mathrm{" + s + "}, \mathrm{Exact}}$" + "$\,(N=$" + str(
                                  N) + ")")
              leg2 = ax.legend([pl_2, pl_3],
                               [
                                   r"$v^{\mathrm{xc}}_{\mathrm{" + s + "}, \mathrm{ML}}$" + "$\,(N=$" + str(N) + ")",
                                   r"$v^{\mathrm{xc}}_{\mathrm{" + s + "}, \mathrm{Exact}}$" + "$\,(N=$" + str(
                                       N) + ")"],
                               loc=(0.50, 1.02), fontsize=12)

          ax.add_artist(leg1)
          plt.savefig("".join([os.getcwd(), test_params["image_dir"]]) +
                      '/DensityPlot_'+test_params["plot_label"]+'.pdf', dpi=900, bbox_inches='tight')
          plt.show()

      if test_params["PlotEnergies"] and (test_params["SystemIdx"].item() == batch_idx):
          fig_height = 5
          fig_width = 5
          fig, axTOT = plt.subplots(figsize=(fig_width, fig_height))
          axTOT.set_axisbelow(True)
          axXC = axTOT.twinx()
          axXC.set_axisbelow(True)
          axTOT.grid(linestyle='--', linewidth=0.6)
          for i in range(2):
              line_tot_k = matplotlib.lines.Line2D([1 + (i + alpha[0].item()), 1 + (i + alpha[-1].item())],
                                                   [E_tot_ref[mix_doubles[i]].detach().cpu().numpy(),
                                                    E_tot_ref[mix_doubles[i] + (
                                                            mix_doubles[1] - mix_doubles[0])].detach().cpu().numpy()],
                                                   color='k', linewidth=1.55, alpha=0.5)

              line_tot_y = matplotlib.lines.Line2D([1 + (i + alpha[0].item()), 1 + (i + alpha[-1].item())],
                                                   [E_tot_ref[mix_doubles[i]].detach().cpu().numpy(),
                                                    E_tot_ref[mix_doubles[i] + (
                                                            mix_doubles[1] - mix_doubles[0])].detach().cpu().numpy()],
                                                   color='y', linewidth=1.5, alpha=0.5)

              axTOT.scatter(x=1 + (i + alpha.detach().cpu().numpy()),
                            y=Etot_mix_arr[i].detach().cpu().numpy(), color='b', s=10, alpha=0.5,
                            label=r'$E^{\mathrm{Tot}}_{\mathrm{ML}}$' if i == 0 else "")
              axXC.scatter(x=1 + (i + alpha.detach().cpu().numpy()),
                           y=Exc_mix_arr[i].detach().cpu().numpy(), color='g', s=10, alpha=0.5,
                           label=r'$E^{\mathrm{xc}}_{\mathrm{ML}}$' if i == 0 else "")


              axTOT.add_line(line_tot_k)
              axTOT.add_line(line_tot_y)


          #axTOT.scatter(x=self.hparams.DimsToTrain, y=E_t_r, color='orange', s=30, alpha=0.9,
          #              label=r'$E^{\mathrm{Tot}}_{\mathrm{Exact}}$', edgecolors='k')
          #axXC.scatter(x=self.hparams.DimsToTrain, y=E_xc_r, color='r', s=30, alpha=0.9,
          #             label=r'$E^{\mathrm{xc}}_{\mathrm{Exact}}$', edgecolors='k')

          axTOT.scatter(x=[1, 2, 3], y=E_t_r, color='y', s=30, alpha=1,
                        label=r'$E^{\mathrm{Tot}}_{\mathrm{Exact}}$', edgecolors='k')
          axXC.scatter(x=[1, 2, 3], y=E_xc_r, color='r', s=30, alpha=0.5,
                      label=r'$E^{\mathrm{xc}}_{\mathrm{Exact}}$', edgecolors='k')
          axXC.legend(fontsize=15, loc=1)
          axTOT.legend(fontsize=15, loc=3)
          axXC.set_xlabel(r'$x\;[a.u.]$', fontsize=20)
          axTOT.set_xlabel(r'$N$', fontsize=20)
          axXC.set_ylabel(r'$E_{xc}  \;[a.u.]$', fontsize=20, color="g")
          axTOT.set_ylabel(r'$E_{tot} \;[a.u.]$', fontsize=20, color="b")
          axTOT.tick_params(axis='y', color='b', which="major", labelcolor="b")
          axXC.tick_params(axis='y', color='g', which="major", labelcolor="g")
          axTOT.set_yticks(axTOT.get_yticks()[::2])
          axXC.set_yticks(axXC.get_yticks()[::2])
          axXC.tick_params(labelsize=15)
          axTOT.tick_params(labelsize=15)
          plt.savefig("".join([os.getcwd(), test_params["image_dir"]]) + '/EnergyCurve_'+
                      test_params["plot_label"]+'.pdf', dpi=900, bbox_inches='tight')
          plt.show()
          fig_height = 6.5
          fig_width = 5
          fig, (axDiff, axDeriv) = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True,
                                                gridspec_kw={'hspace': 0.2})
          axDeriv.plot(N_12arr,
                       np.gradient(E_tot_exact12), color='g', alpha=0.6, linewidth=3,
                       label=r'$\mathrm{d}E^{\mathrm{Tot}}_{\mathrm{Exact}}/\mathrm{d}N$')
          axDeriv.plot(N_23arr[1::], np.gradient(E_tot_exact23[1::]), color='g', linewidth=3, alpha=0.6)
          axDiff.scatter(x=N_12arr,
                         y=Etot_mix_arr[0].detach().cpu().numpy() - E_tot_exact12, color='b', s=10, alpha=0.5,
                         label=r'$E^{\mathrm{Tot}}_{\mathrm{ML}}-E^{\mathrm{Tot}}_{\mathrm{Exact}}$')
          axDiff.scatter(x=N_23arr[1::],
                         y=Etot_mix_arr[1].detach().cpu().numpy()[1::] - E_tot_exact23[1::], color='b', s=10, alpha=0.5)
          axDeriv.scatter(x=N_12arr,
                          y=np.gradient(Etot_mix_arr[0].detach().cpu().numpy()), color='r', s=10, alpha=0.5,
                          label=r'$\mathrm{d}E^{\mathrm{Tot}}_{\mathrm{ML}}/\mathrm{d}N$')
          axDeriv.scatter(x=N_23arr[1::],
                          y=np.gradient(Etot_mix_arr[1].detach().cpu().numpy()[1::]), color='r', s=10, alpha=0.5)

          axDiff.legend(fontsize=15, loc=(0.50, 1.06), framealpha=1)
          axDeriv.legend(fontsize=15, loc=(0, 2.26), framealpha=1)
          shift_diff = 0.01
          axDiff.set_ylim(
              -np.amax(np.abs(np.concatenate((Etot_mix_arr[0].detach().cpu().numpy() - E_tot_exact12, Etot_mix_arr[
                  1].detach().cpu().numpy() - E_tot_exact23)))) - shift_diff,
              np.amax(np.abs(np.concatenate((Etot_mix_arr[0].detach().cpu().numpy() - E_tot_exact12, Etot_mix_arr[
                  1].detach().cpu().numpy() - E_tot_exact23)))) + shift_diff)
          axDeriv.set_xlabel(r'$N$', fontsize=20)
          axDiff.set_ylabel(r'$E^{\mathrm{Tot}}_{\mathrm{ML}}-E^{\mathrm{Tot}}_{\mathrm{Exact}}\;[a.u.]$', fontsize=20,
                            color="b")
          axDeriv.set_ylabel(r'$\mathrm{d}E^{\mathrm{Tot}}_{\mathrm{ML}}/\mathrm{d}N\;[a.u.]$',
                             fontsize=20, color="r")
          axDiff.tick_params(axis='y', color='b', which="major", labelcolor="b")
          axDeriv.tick_params(axis='y', color='r', which="major", labelcolor="r")
          # axDiff.set_yticks(axDiff.get_yticks()[::2])
          # axDeriv.set_yticks(axDeriv.get_yticks()[::2])
          axDeriv.tick_params(labelsize=15)
          axDiff.tick_params(labelsize=15)
          axDiff.grid(linestyle='--', linewidth=0.6, markevery=3)
          axDeriv.grid(linestyle='--', linewidth=0.6, markevery=3)
          # axDeriv.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
          # axDiff.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
          plt.savefig("".join([os.getcwd(), test_params["image_dir"]]) + '/EnergyDerivative_'+
                      test_params["plot_label"]+'.pdf', dpi=900, bbox_inches='tight')
          plt.show()

      if (len(test_params["CompareWith"]) > 0) and (test_params["SystemIdx"].item() == batch_idx):
          import gzip
          import pickle
          import re
          markersize = 5
          # ----------------------------------------------------------
          # Call Energy data of the other models
          # ----------------------------------------------------------
          model_list = [test_params['model_ckpt']]
          if len(test_params["CompareWith"]) > 0:
              model_list += [NNs for NNs in test_params["CompareWith"]]

          Energies_NNs = [None] * len(model_list)
          for n in tqdm(range(len(model_list) - 1)):
              data_file = gzip.open(re.sub('\.ckpt$', '', "".join([os.getcwd(), model_list[n + 1]])) + "_ENERGIES_idx=" +
                                    str(Energies_dic["ExtPotStartIdx"])+"_" + str(test_params["SystemIdx"].item()) +".gz", 'rb')
              Energies_NNs[n + 1] = pickle.load(data_file)
              data_file.close()

          Energies_NNs[0] = Energies_dic
          Energies_dims     = np.array([len(Energies_NNs[i]["Diffs"][0]) for i in range(len(Energies_NNs))])
          if  np.sum(Energies_dims - np.ones(len(Energies_dims))*Energies_dims[0]) != 0:
              raise Exception("Number of fractional points of the models must be the equal!" + "\n" +
                              "".join(["NN " + str(Energies_NNs[i]["plot_label"]) + " FracPoints: " + str(len(Energies_NNs[i]["Diffs"][0])) + "\n"
                                       for i in range(len(Energies_NNs))]))

          E_t_r = E_tot_ref.detach().cpu().numpy()
          fig_height = 6.5
          fig_width = 5
          fig, (axDeriv, axDiff) = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True,
                                                gridspec_kw={'hspace': 0.1})
          E_totExactderiv_contin = np.concatenate((np.gradient(E_tot_exact12), np.gradient(E_tot_exact23[1::])))
          axDeriv.plot(N_12arr, np.gradient(E_tot_exact12),
                       color='k', alpha=1, linewidth=1.5, linestyle="dashed",
                       label=r'$\mathrm{Exact}$')
          axDeriv.plot(N_23arr[1::], np.gradient(E_tot_exact23[1::]),
                       color='k', alpha=1, linewidth=1.5, linestyle="dashed")
          axDeriv.legend(fontsize=15, loc=(0, 2.26), framealpha=1)
          axDiff.set_xlabel(r'$N$', fontsize=20)
          axDiff.set_ylabel(r'$E^{\mathrm{Tot}}_{\mathrm{ML}}-E^{\mathrm{Tot}}_{\mathrm{Exact}}\;[a.u.]$', fontsize=20)
          axDeriv.set_ylabel(r'$\mathrm{d}E^{\mathrm{Tot}}_{\mathrm{ML}}/\mathrm{d}N\;[a.u.]$', fontsize=20)
          axDiff.tick_params(axis='y', which="major")
          axDeriv.tick_params(axis='y', which="major")
          axDeriv.tick_params(labelsize=15)
          axDiff.tick_params(labelsize=15)
          axDiff.grid(linestyle='--', linewidth=0.6, markevery=3)
          axDeriv.grid(linestyle='--', linewidth=0.6, markevery=3)

          e_diff_abs_max  = [None] * len(model_list)
          e_deriv_abs_max = [None] * len(model_list)
          for n in tqdm(range(len(model_list))):
              axDiff.scatter(
                  x=np.concatenate((Energies_NNs[n]["NPoints"][0], Energies_NNs[n]["NPoints"][1]), axis=None),
                  y=np.concatenate((Energies_NNs[n]["Diffs"][0], Energies_NNs[n]["Diffs"][1]), axis=None),
                  s=markersize, alpha=0.7,
                  label=Energies_NNs[n]["plot_label"])
              axDeriv.scatter(
                  x=np.concatenate((Energies_NNs[n]["NPoints"][0], Energies_NNs[n]["NPoints"][1]), axis=None),
                  y=np.concatenate((Energies_NNs[n]["Derivatives"][0], Energies_NNs[n]["Derivatives"][1]), axis=None),
                  s=markersize, alpha=0.7,
                  label=Energies_NNs[n]["plot_label"])
              e_diff_abs_max[n]  = np.amax(
                  np.abs(np.concatenate((Energies_NNs[n]["Diffs"][0], Energies_NNs[n]["Diffs"][1]))))
              e_deriv_abs_max[n] = np.amax(
                  np.abs(np.concatenate((Energies_NNs[n]["Derivatives"][0], Energies_NNs[n]["Derivatives"][1]))-
                         (E_totExactderiv_contin[0] + E_totExactderiv_contin[-1]) / 2.))
          shift_diff  = 0.02
          shift_deriv = 0.02
          axDiff.set_ylim(-np.amax(e_diff_abs_max) - shift_diff, np.amax(e_diff_abs_max) + shift_diff)
          axDeriv.legend(fontsize=15, framealpha=1)
          axDeriv.set_ylim(((E_totExactderiv_contin[0] + E_totExactderiv_contin[-1]) / 2.) - np.amax(e_deriv_abs_max) - shift_deriv,
                           ((E_totExactderiv_contin[0] + E_totExactderiv_contin[-1]) / 2.) + np.amax(e_deriv_abs_max) + shift_deriv)

          plt.savefig("".join([os.getcwd(), test_params["image_dir"]]) + '/EnergyCurveCompare_idx=' +
                                    str(Energies_dic["ExtPotStartIdx"])+"_" + str(test_params["SystemIdx"].item()) +'.pdf', dpi=900, bbox_inches='tight')
          plt.show()
      # ---------------------------------------------------------------------------------------------------------
      return  Exc1_exact,       Exc2_exact,      Exc3_exact,\
              Etot1_exact,      Etot2_exact,     Etot3_exact,\
              Errors1,          Errors2,         Errors3, \
              MSE_Etot,         Var_Etot,        test_params


  def test_epoch_end(self, outputs):
      test_params = outputs[0][-1]
      Errors = []
      for j in range(len(outputs[0][0:-1])):
          Errors.append([outputs[i][j] for i in range(len(outputs))])

      Exc1_exact,     Exc2_exact,     Exc3_exact, \
      Etot1_exact,    Etot2_exact,    Etot3_exact, \
      Errors1,        Errors2,        Errors3, \
      MSE_Etot,       Var_Etot,                      = Errors

      print("____________________________________________________________________________________________________")
      print("ENERGY_ERRORS / INTEGER PARTICLE_NUMBERS in a.u. (" + str(len(Etot1_exact)) + " systems examined):")
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      print("E1_exact:" + " max = " + str(np.round(np.amax(Etot1_exact), 4)) + " | " + "min = " + str(
          np.round(np.amin(Etot1_exact), 4)))
      print("E2_exact:" + " max = " + str(np.round(np.amax(Etot2_exact), 4)) + " | " + "min = " + str(
          np.round(np.amin(Etot2_exact), 4)))
      print("E3_exact:" + " max = " + str(np.round(np.amax(Etot3_exact), 4)) + " | " + "min = " + str(
          np.round(np.amin(Etot3_exact), 4)))
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      print("abs(Etot1_Error)_mean / 1 = ", np.round(np.mean(np.abs(Errors1) / 1.), 4), " | ",
            "abs(Etot1_Error)_max  / 1 = ", np.round(np.amax(np.abs(Errors1) / 1.), 4))
      print("abs(Etot2_Error)_mean / 2 = ", np.round(np.mean(np.abs(Errors2) / 2.), 4), " | ",
            "abs(Etot2_Error)_max  / 2 = ", np.round(np.amax(np.abs(Errors2) / 2.), 4))
      print("abs(Etot3_Error)_mean / 3 = ", np.round(np.mean(np.abs(Errors3) / 3.), 4), " | ",
            "abs(Etot3_Error)_max  / 3 = ", np.round(np.amax(np.abs(Errors3) / 3.), 4))
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      print("abs(Etot1_Error / Exc1_exact)_mean / 1 = ",
            np.round(np.mean(np.abs(Errors1 / np.array(Exc1_exact)) / 1.), 4), " | ",
            "abs(Etot1_Error / Exc1_exact)_max  / 1 = ",
            np.round(np.amax(np.abs(Errors1 / np.array(Exc1_exact)) / 1.), 4))
      print("abs(Etot2_Error / Exc2_exact)_mean / 2 = ",
            np.round(np.mean(np.abs(Errors2 / np.array(Exc2_exact)) / 2.), 4), " | ",
            "abs(Etot2_Error / Exc2_exact)_max  / 2 = ",
            np.round(np.amax(np.abs(Errors2 / np.array(Exc2_exact)) / 2.), 4))
      print("abs(Etot3_Error / Exc3_exact)_mean / 3 = ",
            np.round(np.mean(np.abs(Errors3 / np.array(Exc3_exact)) / 3.), 4), " | ",
            "abs(Etot3_Error / Exc3_exact)_max  / 3 = ",
            np.round(np.amax(np.abs(Errors3 / np.array(Exc3_exact)) / 3.), 4))
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      print("____________________________________________________________________________________________________")
      print("ENERGY_ERRORS / FRACTIONAL PARTICLE_NUMBERS in a.u. (" + str(len(Etot1_exact)) + " systems examined):")
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      print("Var_Etot_mean = ", np.round(np.mean(Var_Etot), 4))
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      print("MSE_Etot_mean = ", np.round(np.mean(MSE_Etot), 4))
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

      self.log("Var_Etot_mean" , np.mean(Var_Etot))
      self.log("MSE_Etot_mean",  np.mean(MSE_Etot))

      if len(test_params["H2Dissociation"]) > 0:
          self.DissociationCurve(test_params)

      if test_params["PlotVxcFrac"] != -1:
          self.Vxc_Jump_Prediction(test_params)

  def DissociationCurve(self, test_params):
      import matplotlib
      from matplotlib import pyplot as plt
      import pandas as pd
      import gzip
      import pickle
      import re
      # ------------------------------------------------
      # Latex Formatting
      # ------------------------------------------------
      matplotlib.rcParams['mathtext.fontset'] = 'cm'
      matplotlib.rcParams['font.family'] = 'STIXGeneral'
      # ------------------------------------------------
      def floatArray(string):
          return np.array(string.replace('[', '').replace(']', '')
                          .replace(',', ' ').replace("'", ' ').split()).astype(float)

      model_list = [test_params['model_ckpt']]
      if len(test_params["CompareWith"]) > 0:
          model_list += [NNs for NNs in test_params["CompareWith"]]

      # ----------------------------------------------------------
      # Call H2 data of the other models
      # ----------------------------------------------------------
      H2_NNs = [None] * len(model_list)
      for n in tqdm(range(len(model_list) - 1)):
          data_file = gzip.open(re.sub('\.ckpt$', '', "".join([os.getcwd(), model_list[n + 1]])) + "_H2DISSCOCIATION.gz", 'rb')
          H2_NNs[n + 1] = pickle.load(data_file)
          data_file.close()
      # ----------------------------------------------------------
      # Read exact H2 data from file
      # ----------------------------------------------------------
      H2_Exact = pd.read_csv("".join([os.getcwd(), test_params["H2Dissociation"]]))
      H2_ExactEnergies = H2_Exact["Energy"].to_numpy()
      data_lenght = len(H2_ExactEnergies)
      min_dist = 2 * floatArray(H2_Exact["nucs_array"].iloc[0])[-1]
      max_dist = 2 * floatArray(H2_Exact["nucs_array"].iloc[-1])[-1]
      fig_height = 4
      fig_width = 5
      fH2, axH2 = plt.subplots(1, sharex=True, sharey=True)
      fH2.set_figheight(fig_height)
      fH2.set_figwidth(fig_width)
      plt.scatter(np.linspace(min_dist, max_dist, data_lenght), H2_ExactEnergies, color="k", s=10, alpha=0.7, label="Exact")
      plt.xlabel("Distance [a.u.]", fontsize=20)
      plt.tick_params(labelsize=15)
      plt.xlim(0, 10)
      plt.ylabel("Energy [a.u.]", fontsize=20)
      plt.tick_params(labelsize=13)
      axH2.grid(linestyle='--', linewidth=0.6)
      # ______________________________________
      H2_ExactEnergies = torch.from_numpy(H2_ExactEnergies)
      H2_NNEnergies = [None] * len(H2_ExactEnergies)

      # ----------------------------------------------------------
      # Compute H2 energies (as NN prediction)
      # ----------------------------------------------------------
      print("\nComputing H_2 Dissociation...")
      for i in tqdm(range(len(H2_ExactEnergies))):
          NucNucInteraction = 1. / np.sqrt((1. + (i * 0.1) ** 2))  # [a.u.]
          points = torch.from_numpy(floatArray(H2_Exact['points'][i])).double()
          dx = points[1] - points[0]
          v_ext_H2 = torch.from_numpy(floatArray(H2_Exact['v_ext'][i])).double()
          Dens_tot_H2 = torch.from_numpy(floatArray(H2_Exact['Dens_data_total'][i])).double().view(1, 1, -1)
          Dens_up_H2 = torch.from_numpy(floatArray(H2_Exact['Dens_data_up'][i])).double().view(1, 1, -1)
          Dens_down_H2 = torch.from_numpy(floatArray(H2_Exact['Dens_data_down'][i])).double().view(1, 1, -1)
          DensUpAndDown_H2 = torch.cat((Dens_up_H2, Dens_down_H2), 1)
          Dens_H2 = DensUpAndDown_H2 if self.hparams.Spin else Dens_tot_H2
          if test_params["gpus_num"] != 0 or len(test_params["gpus_devices"]) != 0:
              Dens_H2 = Dens_H2.cuda()
          Dens_H2.requires_grad = True
          E_xc_H2 = self((points.unsqueeze(0), Dens_H2))
          V_xc_H2 = \
              torch.autograd.grad(inputs=Dens_H2, outputs=E_xc_H2, create_graph=False,
                                  retain_graph=True, grad_outputs=torch.ones_like(E_xc_H2))[0] / dx
          Dens_H2 = Dens_H2.squeeze(0)
          E_xc_H2 = E_xc_H2.squeeze(0)
          V_xc_H2 = V_xc_H2.squeeze(0)
          # ---------------------------------------------------------------------------------------------------------
          # COMPUTE TOTAL ENERGY
          # ---------------------------------------------------------------------------------------------------------
          KSsystem = {
              "dim":       2,
              "v_ext":     v_ext_H2,
              "points":    points,
              "N":         len(points),
              "dx":        dx.item(),
              'laplOrder': 4,
              "occ":       None,
              "up_occ":    None,
              "down_occ":  None,
              }
          KSsystem['occ'], KSsystem['up_occ'], KSsystem['down_occ'] = self.GS_dens_splitter(KSsystem['dim'])
          Dens_KS, E_tot_NN = self.KSIterations(KSsystem=KSsystem, Dens_inp=Dens_H2, V_xc_inp=V_xc_H2, E_xc=E_xc_H2)
          H2_NNEnergies[i] = E_tot_NN.detach().cpu().numpy().item() + NucNucInteraction

      H2_NNs[0] = {
          'model_ckpt':  test_params['model_ckpt'],
          "plot_label":  test_params["plot_label"],
          "Energies":    H2_NNEnergies,
      }

      H2_NN_file = gzip.open(re.sub('\.ckpt$', '', "".join([os.getcwd(), test_params['model_ckpt']])) + "_H2DISSCOCIATION.gz", 'wb')
      pickle.dump(H2_NNs[0], H2_NN_file)
      H2_NN_file.close()

      for n in tqdm(range(len(model_list))):
          plt.scatter(np.linspace(min_dist, max_dist, data_lenght), H2_NNs[n]["Energies"], s=10, alpha=0.7,
                      label=H2_NNs[n]["plot_label"])

      plt.legend(fontsize=15)

      plt.savefig("".join([os.getcwd(), test_params["image_dir"]]) + '/H2Dissociation_'+
                  test_params["plot_label"]+'.pdf', dpi=900, bbox_inches='tight')
      plt.show()

  def Vxc_Jump_Prediction(self, test_params):
      import matplotlib
      from matplotlib import pyplot as plt
      import pandas as pd
      matplotlib.rcParams['mathtext.fontset'] = 'cm'
      matplotlib.rcParams['font.family'] = 'STIXGeneral'
      # ------------------------------------------------
      def floatArray(string):
          return np.array(string.replace('[', '').replace(']', '').replace(',', ' ').split()).astype(float)
      # ------------------------------------------------
      N = test_params["PlotVxcFrac"].item()
      frac = N - int(N)
      Set_exact = pd.read_csv("".join([os.getcwd(), test_params["VxcFracFile"]]))
      Set_exact_int = Set_exact[(Set_exact["dim"] == N - frac)].reset_index(drop=True)
      Set_exact_frac = Set_exact[(Set_exact["dim"] == N)].reset_index(drop=True)
      points = torch.from_numpy(floatArray(Set_exact_int['points'][0])).double()
      dx = points[1] - points[0]

      i = test_params["SystemIdx"].item()
      MSE_tot_int,  MSE_up_int,  MSE_down_int =  Set_exact_int["MSE"][i], \
                                                 Set_exact_int["MSE_up"][i], \
                                                 Set_exact_int["MSE_down"][i]
      MSE_tot_frac, MSE_up_frac, MSE_down_frac = Set_exact_frac["MSE"][i], \
                                                 Set_exact_frac["MSE_up"][i], \
                                                 Set_exact_frac["MSE_down"][i]

      MSEs = np.array([MSE_tot_int, MSE_up_int,  MSE_down_int, MSE_tot_frac, MSE_up_frac, MSE_down_frac]).astype(float)

      if len(np.where(MSEs > 1.5*1e-8)[0]) > 0:
          raise Exception("MSE > 1.5*1e-8 -> ", MSEs)

      Dens_tot_int = torch.from_numpy(floatArray(Set_exact_int['Dens_data_total'][i])).double().view(1, 1, -1)
      Dens_tot_frac = torch.from_numpy(floatArray(Set_exact_frac['Dens_data_total'][i])).double().view(1, 1, -1)
      Dens_up_int = torch.from_numpy(floatArray(Set_exact_int['Dens_data_up'][i])).double().view(1, 1, -1)
      Dens_up_frac = torch.from_numpy(floatArray(Set_exact_frac['Dens_data_up'][i])).double().view(1, 1, -1)
      Dens_down_int = torch.from_numpy(floatArray(Set_exact_int['Dens_data_down'][i])).double().view(1, 1, -1)
      Dens_down_frac = torch.from_numpy(floatArray(Set_exact_frac['Dens_data_down'][i])).double().view(1, 1, -1)
      
      #---fix NORMALIZATION -----
      Dens_tot_int  = (Dens_tot_int  / (Dens_tot_int.sum() * dx)) * int(N - frac)
      Dens_tot_frac = (Dens_tot_frac / (Dens_tot_frac.sum() * dx)) * N
      #---------------------------
      
      DensUpAndDown_int = torch.cat((Dens_up_int, Dens_down_int), 1)
      DensUpAndDown_frac = torch.cat((Dens_up_frac, Dens_down_frac), 1)
      Dens_int = DensUpAndDown_int if self.hparams.Spin else Dens_tot_int
      Dens_frac = DensUpAndDown_frac if self.hparams.Spin else Dens_tot_frac
      if test_params["gpus_num"] != 0 or len(test_params["gpus_devices"]) != 0:
          Dens_frac = Dens_frac.cuda()
          Dens_int  = Dens_int.cuda()
      Dens_int.requires_grad = True
      Dens_frac.requires_grad = True
      E_xc_int = self((points.unsqueeze(0), Dens_int))
      E_xc_frac = self((points.unsqueeze(0), Dens_frac))
      V_xc_int = \
          torch.autograd.grad(inputs=Dens_int, outputs=E_xc_int, create_graph=False,
                              retain_graph=True, grad_outputs=torch.ones_like(E_xc_int))[0] / dx
      V_xc_frac = \
          torch.autograd.grad(inputs=Dens_frac, outputs=E_xc_frac, create_graph=False,
                              retain_graph=True, grad_outputs=torch.ones_like(E_xc_frac))[0] / dx
      if self.hparams.Spin:
          V_xc_int_exact = [floatArray(Set_exact_int['v_xc_up'][i]), floatArray(Set_exact_int['v_xc_down'][i])]
          V_xc_frac_exact = [floatArray(Set_exact_frac['v_xc_up'][i]),
                             floatArray(Set_exact_frac['v_xc_down'][i])]
      else:
          V_xc_int_exact = [floatArray(Set_exact_int['v_xc'][i])]
          V_xc_frac_exact = [floatArray(Set_exact_frac['v_xc'][i])]

      Dens_int = Dens_int.squeeze(0);
      Dens_frac = Dens_frac.squeeze(0)
      V_xc_int = V_xc_int.squeeze(0);
      V_xc_frac = V_xc_frac.squeeze(0)

      fig_height = 4
      fig_width = 5
      fvxc, ax = plt.subplots(1, sharex=True, sharey=True)
      fvxc.set_figheight(fig_height)
      fvxc.set_figwidth(fig_width)
      plt.xlabel(r"$x\;[a.u.]$", fontsize=20)
      plt.tick_params(labelsize=15)
      plt.ylabel(r'$\rho, v^{\mathrm{xc}} \;[a.u.]$', fontsize=20)
      plt.tick_params(labelsize=13)
      ax.grid(linestyle='--', linewidth=0.6)
      SpinPol = 0  # 0 == up, 1 == down
      s = "tot"
      if self.hparams.Spin:
          s = "down" if SpinPol else "up"
      else:
          SpinPol = 0
      linew = 2.5
      ax.plot(points, Dens_int[SpinPol].detach().cpu().numpy(), alpha=0.4, color="k", linewidth=1,
              label=r"$\rho_{\mathrm{" + s + "}}$" + "$\,(N=$" + str(N - frac) + ")")
      ax.plot(points, V_xc_int[SpinPol].detach().cpu().numpy(), alpha=0.4, color="r", linewidth=linew,
              label=r"$v^{\mathrm{xc}}_{\mathrm{" + s + "}, \mathrm{ML}}$" + "$\,(N=$" + str(N - frac) + ")")
      ax.plot(points, V_xc_int_exact[SpinPol], alpha=0.4, color="b", linewidth=linew / 2.,
              linestyle="dashed",
              label=r"$v^{\mathrm{xc}}_{\mathrm{" + s + "}, \mathrm{Exact}}$" + "$\,(N=$" + str(N - frac) + ")")
      pl_1, = ax.plot(points, Dens_frac[SpinPol].detach().cpu().numpy(), linewidth=1, linestyle="dashed", color="k")
      pl_2, = ax.plot(points, V_xc_frac[SpinPol].detach().cpu().numpy(), linewidth=linew, color="r", )
      pl_3, = ax.plot(points, V_xc_frac_exact[SpinPol], linewidth=linew / 2., color="b", linestyle="dashed")

      leg1 = ax.legend(loc=(0., 1.02), fontsize=12)
      leg2 = ax.legend([pl_1, pl_2, pl_3],
                       [r"$\rho_{\mathrm{" + s + "}}$" + "$\,(N=$" + str(N) + ")",
                        r"$v^{\mathrm{xc}}_{\mathrm{" + s + "}, \mathrm{ML}}$" + "$\,(N=$" + str(N) + ")",
                        r"$v^{\mathrm{xc}}_{\mathrm{" + s + "}, \mathrm{Exact}}$" + "$\,(N=$" + str(N) + ")"],
                       loc=(0.50, 1.02), fontsize=12)
      ax.add_artist(leg1)
      plt.xlim(points[0], points[-1])

      plt.savefig("".join([os.getcwd(), test_params["image_dir"]]) + '/VxcJump_'+test_params["plot_label"]+'.pdf',
                  dpi=900, bbox_inches='tight')
      plt.show()

  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
      scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.hparams.learning_rate,
                                                   max_lr=15 * self.hparams.learning_rate, step_size_up=115000,
                                                   step_size_down=115000, gamma=0.999, cycle_momentum=False)
      return optimizer

  def validation_epoch_end(self, outputs):
    global time_convergence
    avg_loss = torch.stack(
    [x["val_loss"] for x in outputs]).mean()
    self.log("ptl/val_loss", avg_loss)

  @staticmethod
  def add_model_specific_args():
      """
      Parameters you define here will be available to your model through self.hparams
      """
      def strToArray(string, type):
          return np.array(string.replace('[', '').replace(']', '').replace(',', ' ').split()).astype(type)

      def floatArray(string): return strToArray(string, float)
      def intArray(string):   return strToArray(string, int)


      parser = ArgumentParser(fromfile_prefix_chars='@', add_help=True)

      parser.add_argument('--batch-size', '--batch_size', '--batchsize',
                          required=True,
                          type=int,
                          help="batch size used for the training"
                          )
      parser.add_argument('--data_dir', '--data-dir',
                          type=str,
                          required=True,
                          help='Data used for the training'
                          )
      parser.add_argument('--DimsToTrain',
                          type=floatArray,
                          required=True,
                          default="[1,2,3]",
                          help='Array of dimensions which will be used for the training. It must '
                               'contain at least the integer (1,2,3) densities. If fractionals desnities .x used, '
                               'they must appear as 1.x and 2.x in DimsToTrain.'

                          )
      parser.add_argument('--Spin',
                          action="store_true",
                          help="Spin densities will be used"
                          )
      parser.add_argument('--idxs_ExtPotsTrain',
                          type=intArray,
                          required=True,
                          help="1D array containg start and end index of the arrangement "
                               "of external potential for trainings sets"
                          )
      parser.add_argument('--idxs_ExtPotsVal',
                          required=True,
                          type=intArray,
                          help="1D array containg start and end index of the arrangement "
                               "of external potential for validation sets"
                          )
      parser.add_argument('--idxs_ExtPotsTest',
                          type=intArray,
                          default="[0,0]",
                          help="1D array containg start and end index of the arrangement "
                               "of external potential for test sets. This Option is NOT"
                               "necessary for the training!"
                          )
      parser.add_argument('--Disc',
                          action="store_true",
                          help="Non-differnetiable auxiliary function will be implemented"
                          )
      parser.add_argument('--kernelsize',
                          type=int,
                          default=9,
                          help="== scan window size, if WindowPlusLDA==True, otherwise "
                               "len(Conv_OutChannels) * kernelsize == scan window size"
                          )
      parser.add_argument('--WindowPlusLDA',
                          action="store_true",
                          help="Jonathan's scanning method"
                          )
      parser.add_argument('--LDA_in_channels',
                          type=int,
                          default=16,
                          help="Out channels of scan window, if WindowPlusLDA used"
                          )
      parser.add_argument('--LDA_LayerOutDims',
                          type=intArray,
                          default="[16 16 16 16 1]",
                          help="array containing out dimensions of each linear layer"
                          )
      parser.add_argument('--Conv_OutChannels',
                          type=intArray,
                          default="[16 16 16 1]",
                          help="Array containing out dimensions of each convolutional layer",
                          )
      parser.add_argument('--gpus_num',
                          type=int,
                          default=0,
                          help="Specify number of GPUs to use"
                          )
      parser.add_argument('--gpus_devices',
                          type=intArray,
                          default="[]",
                          help="Specify which GPUs to use (don't use when running on cluster)"
                          )
      parser.add_argument("--num_workers",
                          default=0,
                          type=int,
                          help="Number of data loading workers (default: 0), crashes on some machines if used"
                          )
      parser.add_argument("--epochs",
                          default=390,
                          type=int,
                          required=True,
                          metavar="N",
                          help="Number of total epochs to run"
                          )
      parser.add_argument("--optim",
                          default="AdamW",
                          type=str,
                          metavar="str",
                          help="Choose an optimizer; SGD, Adam or AdamW"
                          )
      parser.add_argument("--learning_rate", "--learning-rate", "--lr",
                          default=3e-4,
                          type=float,
                          metavar="float",
                          help="Initial learning rate (default: 3e-4)"
                          )
      parser.add_argument("--continue_ckpt",
                         type=str,
                         help="If path given, model from checkpoint will continue to be trained"
                         )
      parser.add_argument("--continue_hparams",
                          type=str,
                          help="path to hparams.yaml used if continue_ckpt is given"
                          )
      parser.add_argument("--train_jump",
                          action="store_true",
                          help="XC_jump will be included in loss function"
                          )
      parser.add_argument("--SpinMirror",
                          action="store_true",
                          help="Spin channels will swapped after each training iteration "
                               "and used for training additionally"
                          )
      args = parser.parse_args()
      mid_idx=int(len(args.DimsToTrain) / 2.)

      if not args.Spin and args.SpinMirror:
          raise Exception("SpinMirror selected, but not Spin!")

      for d in range(len(args.DimsToTrain)):
          if args.DimsToTrain[d] + 1 not in args.DimsToTrain:
              raise Exception("Wrong declaration of DimsToTrain array")
          if d == mid_idx: break

      if args.train_jump:
          if len(args.DimsToTrain) != args.batch_size:
              raise Exception("len(DimsToTrain) must be equal to batch_size!")
          if args.Spin:
              raise Exception("train_jump not available for Spin yet")

      return parser

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import cg
import findiff
from scipy.optimize import minimize
import sys

def Orbitals(P, v_eff_diag, occ):
    # --------------------------------------------------------------------------------------------------------------
    # SOLVE EIGENVALUE PROBLEM
    # --------------------------------------------------------------------------------------------------------------
    T = KineticOP(P)
    V_eff = diags(v_eff_diag, 0).toarray()
    Ham = T + V_eff
    Evals, Evecs = eigh(Ham)
    idx = Evals.argsort()
    Evals = Evals[idx]
    Evecs = Evecs[:, idx]
    # --------------------------------------------------------------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------------------------------------------------------------
    dens_M = np.multiply(np.conjugate(Evecs), Evecs)
    C_norm = 1. / np.sqrt((dens_M.sum(axis=0) * P['dx']))
    dens_M = np.multiply(C_norm ** 2, dens_M)
    Evecs = np.multiply(C_norm, Evecs)

    return Evals[0:len(occ)], Evecs[:, 0:len(occ)], dens_M[:, 0:len(occ)]


def Density(P, D_arr, occ):
    D = np.sum(np.multiply(occ, D_arr), axis=1)
    return D

def V_Hartree(P, Dens):
    x, y = np.meshgrid(P['points'], P['points'])
    f_Clmb = np.multiply(Dens, (1. / np.sqrt(1 + (y - x) ** 2)).T).T
    V_Clmb = f_Clmb.sum(axis=0) * P['dx']
    return V_Clmb

def V_HartreeTorch(P, Dens):
    import torch
    x, y = torch.meshgrid(P['points'], P['points'])
    f_Clmb = torch.multiply(Dens, (1. / torch.sqrt(1 + (y - x) ** 2)).T).T
    V_Clmb = f_Clmb.sum(axis=0) * P['dx']
    return V_Clmb

def Inversion(P, occ, Dens_spin):
    v_H_diag = V_Hartree(P, P['Dens_data_total'])#, occ = occ)
    v_ext_diag = P['v_ext']

    def Loss_fun(z):
        # --------------------------------------------------------------------------------------------------------------
        # COMPUTE APPROX. DENS
        # --------------------------------------------------------------------------------------------------------------
        v_eff_diag = z + v_H_diag + v_ext_diag
        Loss_fun.evals, Loss_fun.Psi, D_Matrix = Orbitals(P, v_eff_diag, occ)
        Loss_fun.Dens = Density(P, D_Matrix, occ)

        return P['dx'] * np.sum(np.multiply(P['weight'], (Dens_spin - Loss_fun.Dens) ** 2))

    def gradLoss_xc(z):
        # --------------------------------------------------------------------------------------------------------------
        # COMPUTE APPROX. DENS
        # --------------------------------------------------------------------------------------------------------------
        v_eff_diag = z + v_H_diag + v_ext_diag
        evals, Psi, D_Matrix = Orbitals(P, v_eff_diag, occ)
        Dens = Density(P, D_Matrix, occ)
        Ham = KineticOP(P) + diags(v_eff_diag, 0).toarray()
        # --------------------------------------------------------------------------------------------------------------
        # DIRICHLET CONDITION FOR P'S
        # --------------------------------------------------------------------------------------------------------------
        if P['dirichlet'] != None:
            bound_l, bound_r = 0, len(Dens_spin)
            l_tofind, r_tofind = True, True
            cond = P['dirichlet']

            for k in range(len(Dens_spin)):
                if Dens_spin[k] >= cond and l_tofind:
                    bound_l = k
                    l_tofind = False
                if Dens_spin[len(Dens_spin) - 1 - k] >= cond and r_tofind:
                    bound_r = len(Dens_spin) - k
                    r_tofind = False
        # --------------------------------------------------------------------------------------------------------------
        # COMPUTE P'S
        # --------------------------------------------------------------------------------------------------------------
        grad = 0.

        for i in range(len(evals)):
            A_0 = Ham - diags(evals[i] * np.ones(P['N']), 0).toarray()
            c = occ[i] * 4 * np.multiply(np.multiply(P['weight'], (Dens_spin - Dens)), Psi[:, i])
            #if (occ.sum() % 2 != 0) and (i == len(evals) - 1):
            #    c = 0.5 * c # all orbitals single occ

            c = c - P['dx'] * Psi[:, i].dot(c) * Psi[:, i]
            p = cg(A_0, c)[0]

            if P['dirichlet'] != None:
                p[0:bound_l] = 0
                p[bound_r:len(z)] = 0

            grad += np.multiply(p, Psi[:, i])

        return grad

    Loss_fun_min = minimize(Loss_fun, x0=P['v_xc_init'], jac=gradLoss_xc, method='CG',tol=1e-11, options={'gtol': 1e-12})
    MSE = Loss_fun_min.fun
    max_Err = np.amax(np.max(Loss_fun.Dens - Dens_spin))
    v_xc_inv = Loss_fun_min.x

    return Loss_fun.evals, Loss_fun.Psi, Loss_fun.Dens, v_xc_inv, v_H_diag, MSE, max_Err


def KineticOP(P):
    lapl_fidiff = findiff.coefficients(deriv=2, acc=P['laplOrder'])['center']
    coeff_lapl  = lapl_fidiff['coefficients']
    pos_lapl    = lapl_fidiff['offsets']
    T = -0.5 * diags(coeff_lapl, pos_lapl, shape=(P['N'], P['N'])).toarray() / (P['dx'] ** 2)
    return T


def E_kinetic(P, Psi, occ):
    #print("occ.sum(): ", occ.sum(), "Psi.shape[-1]: ", Psi.shape[-1], "occ: ", str(occ), )
    return  P['dx'] * np.multiply(np.conjugate(Psi), np.multiply(occ, np.matmul(KineticOP(P), Psi))).sum()

def Energies(P, Dens, v_ext, v_H):
    E_ext = P['dx'] * np.multiply(v_ext, Dens).sum()
    E_H   = 0.5 * P['dx'] * np.multiply(Dens, v_H).sum()
    return E_ext, E_H


def Energies2(P, Dens, evals, v_xc, v_H, occ):
    #print("occ.sum(): ", occ.sum(), "Psi.shape[-1]: ", evals.shape[-1], "occ: ", str(occ), )
    E_eigen =  np.multiply(occ, evals).sum()
    E_vxc   =  P['dx'] * np.multiply(Dens, v_xc).sum()
    E_H     = 0.5 * P['dx'] * np.multiply(Dens, v_H).sum()

    return E_eigen, E_vxc, E_H
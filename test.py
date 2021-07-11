import os
import pytorch_lightning as pl
from XC_Model import XC_Model
from argparse import ArgumentParser
from XC_DataModule import XC_DataModule
import numpy as np
import sys

def intArray(string):
    return np.array(string.replace('[', '').replace(']', '').replace(',', ' ').split()).astype(int)

parser = ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument("--model_ckpt",
                    type=str,
                    help="If path given, model from checkpoint will continue to be tested"
                    )

parser.add_argument("--num_workers",
                    default=2,
                    type=int,
                    help="Number of data loading workers (default: 2), crashes on some machines if used"
                    )

parser.add_argument('--PlotEnergies',
                    action="store_true",
                    help="Plot total and exchange correlation energy of (FracPoints) densities"
                    )

parser.add_argument('--FracPoints',
                    type=int,
                    default=4,
                    help="Number of (fractional) densities per integer interval (needs to be > 2). \n"
                         "They will be use to plot the energies, if 'PlotEnergies' is selected and to "
                         "quantify the total energy error."
                    )

parser.add_argument('--PlotDensitiesDim',
                    default=-1,
                    type=float,
                    help="Fractional density to plot, if --PlotDensities is selected"
                    )

parser.add_argument('--PlotVxcFrac',
                    default=-1,
                    type=float,
                    help="Fractional V_xc to plot, if 'PlotVxc' is selected"
                    )

parser.add_argument('--VxcFracFile',
                    default="",
                    type=str,
                    help="File (csv format) used for 'PlotVxcFrac'"
                    )

parser.add_argument('--H2Dissociation',
                    type=str,
                    default="",
                    help="Path to the (exact) H2 dissociation file (csv format).\n"
                         "Dissociation energies of H_2 predicted by NN will be plotted"
                    )

parser.add_argument('--CompareWith',
                    default="",
                    type=str,
                    help="List of other models (checkpoints) to compare"
                    )

parser.add_argument('--SystemIdx',
                    default=0,
                    type=int,
                    help="batch index of system used if 'PlotEnergies' is selected "
                         "and 'CompareWith' is not empty, as well as used for all plots"
                    )

parser.add_argument('--plot_label',
                    default="NN",
                    type=str,
                    help="plot label for the model"
                    )

parser.add_argument('--idxs_ExtPotsTest',
                    type=intArray,
                    required=True,
                    default="[700,703]",
                    help="1D array containg start and end index of the arrangement "
                         "of external potential for test sets"
                    )

parser.add_argument('--image_dir',
                    type=str,
                    default=os.getcwd(),
                    help="Folder where pictures will be saved"
                    )
parser.add_argument('--gpus',
                    type=int,
                    default=0,
                    help="Specify which GPUs to use (don't use when running on cluster)"
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
parser.add_argument('--data_dir', '--data-dir',
                          type=str,
                          help='Data used for the training'
                          )

args = parser.parse_args()

model = XC_Model.load_from_checkpoint(os.getcwd() + args.model_ckpt)

if args.PlotDensitiesDim not in model.hparams.DimsToTrain and args.PlotDensitiesDim != -1:
    raise Exception("'DimsToTrain' (" + str(model.hparams.DimsToTrain) +") does not contain 'PlotDensitiesDim'!")
if args.FracPoints < 3:
    raise Exception("'FracPoints' must be > 2")

print("HYPERPARAMETERS:\n", model.hparams)

#if not hasattr(model.hparams, "train_jump"):

gpus_attr = args.gpus_num \
    if len(args.gpus_devices) == 0 \
    else list(args.gpus_devices)


trainer = pl.Trainer(precision=64,
                     checkpoint_callback=False,
                     logger=None,
                     gpus=gpus_attr,
                     )

dm = XC_DataModule(data_dir=args.data_dir,
                   batch_size=model.hparams.batch_size,
                   DimsToTrain=model.hparams.DimsToTrain,
                   idxs_ExtPots=[model.hparams.idxs_ExtPotsTrain,
                                 model.hparams.idxs_ExtPotsVal,
                                 args.idxs_ExtPotsTest],
                   CheckIntegError_Str="",
                   num_workers=args.num_workers,
                   test_params=vars(args),
                   )
dm.prepare_data()
dm.setup("test")
trainer.test(model, test_dataloaders=dm.test_dataloader())
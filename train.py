import os
import sys
import torch
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from XC_Model import XC_Model
from XC_DataModule import XC_DataModule
import numpy as np
import datetime

def main(args):

    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    if args.continue_ckpt == None:
        model = XC_Model(**vars(args))
    else:
        model = XC_Model.load_from_checkpoint(os.getcwd() + args.continue_ckpt,
                                              hparams_file=os.getcwd() + args.continue_hparams)
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    name = "runs/f-{s}_t-" + "{date:%d-%m-%Y_%H:%M:%S}".format(
        date=datetime.datetime.now())
    name += "_prefix:" + sys.argv[1].replace("/", "|")
    name += '_Disc:' + str(model.hparams.Disc)
    name += '_Spin:' + str(model.hparams.Spin)
    name +=  '_KS:' + str(model.hparams.kernelsize)
    name += '_WPLDA:' + str(model.hparams.WindowPlusLDA)
    name += '_L:' + ''.join([str(el) + '_' for el in model.hparams.LDA_LayerOutDims])
    name += '_Tridx:' + ''.join([str(el) + '_' for el in model.hparams.idxs_ExtPotsTrain])
    name += '_DtoT:' + ''.join([str(el) + '_' for el in np.union1d(model.hparams.DimsToTrain, args.DimsToTrain)])
    if hasattr(model.hparams, "SpinMirror"): name += 'SpMirr:' + str(model.hparams.SpinMirror)
    if hasattr(model.hparams, "train_jump"): name += 'trJump:' + str(model.hparams.train_jump)

    logger = TensorBoardLogger("tb_logs", name=name)
    gpus_attr = args.gpus_num \
        if len(args.gpus_devices) == 0 \
        else list(args.gpus_devices)

    if args.continue_ckpt != None: model.hparams.epochs = args.epochs
    #print(gpus_attr)
    #sys.exit()
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val_loss:.5f}',
        dirpath=os.path.join(os.getcwd(), 'tb_logs/', name),
        save_top_k=10,
        verbose=True,
        monitor='val_loss',
        mode='min')

    trainer = pl.Trainer(
        max_epochs=model.hparams.epochs,
        gpus=gpus_attr,
        precision=64,
        callbacks=[checkpoint_callback],
        logger=logger,
        check_val_every_n_epoch=2,
    )

    #----- new train_jump option ------
    train_jump_attr = False
    if hasattr(model.hparams, "train_jump"):
        train_jump_attr = model.hparams.train_jump

    # ---------------------------------
    print(model.hparams.DimsToTrain)
    print(model.hparams.data_dir)
    print(model.hparams.idxs_ExtPotsTrain)
    print(model.hparams.idxs_ExtPotsVal)

    if args.continue_ckpt != None: # !!!!!!!!!!!!!
        model.hparams.DimsToTrain       = args.DimsToTrain
        model.hparams.data_dir          = args.data_dir
        model.hparams.idxs_ExtPotsTrain = args.idxs_ExtPotsTrain
        model.hparams.idxs_ExtPotsVal   = args.idxs_ExtPotsVal
        model.hparams.num_workers       = args.num_workers
        model.hparams.continue_ckpt     = args.continue_ckpt

    print("-----------------------------------")
    print(model.hparams.DimsToTrain)
    print(model.hparams.data_dir)
    print(model.hparams.idxs_ExtPotsTrain)
    print(model.hparams.idxs_ExtPotsVal)
    # ---------------------------------

    dm = XC_DataModule(data_dir=model.hparams.data_dir,
                       train_jump=train_jump_attr,
                       batch_size=model.hparams.batch_size,
                       DimsToTrain=model.hparams.DimsToTrain ,
                       idxs_ExtPots=[model.hparams.idxs_ExtPotsTrain,
                                     model.hparams.idxs_ExtPotsVal,
                                     model.hparams.idxs_ExtPotsTest],
                       CheckIntegError_Str="",
                       num_workers=model.hparams.num_workers
                       )
    print("HYPERPARAMETERS:\n", model.hparams)
    print("Cuda available: ", torch.cuda.is_available(), "\n")
    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, dm)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    model_parser = XC_Model.add_model_specific_args()
    hyperparams = model_parser.parse_args()
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)

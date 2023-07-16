import os

# models
from models.nerf import *
from models.rendering import *
from models.networks import E_attr

# optimizer, scheduler, visualization
from utils import *

# metrics
from utils.metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from models.nerfsystem import NeRFSystem


def run_train(hparams):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # torch.multiprocessing.set_start_method('spawn')
    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(filepath=os.path.join(hparams.save_dir,
                                              f'ckpts/{hparams.exp_name}', '{epoch:d}'),
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=-1)

    logger = TestTubeLogger(save_dir=os.path.join(hparams.save_dir,"logs"),
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    if hparams.continue_train_semantic and hparams.enable_semantic:

        enc_a = E_attr(3, hparams.N_a).cuda()
        load_ckpt(enc_a, hparams.ckpt_path, model_name='enc_a')

        system.enc_a = enc_a
        system.models_to_train[0] = enc_a

        nerf_coarse = NeRF('coarse',
                                enable_semantic=hparams.enable_semantic,
                                num_semantic_classes=hparams.num_semantic_classes,
                                is_test=True,
                                in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                in_channels_dir=6 * hparams.N_emb_dir + 3)

        nerf_fine = NeRF('fine',
                                  enable_semantic=hparams.enable_semantic,
                                  num_semantic_classes=hparams.num_semantic_classes,
                                  in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                  in_channels_dir=6 * hparams.N_emb_dir + 3,
                                  encode_appearance=hparams.encode_a,
                                  in_channels_a=hparams.N_a,
                                  is_test=True,
                                  encode_random=hparams.encode_random)


        load_ckpt(nerf_coarse, hparams.ckpt_path, model_name='nerf_coarse')
        load_ckpt(nerf_fine, hparams.ckpt_path, model_name='nerf_fine')


        system.models['coarse'] = nerf_coarse
        system.models['fine'] = nerf_fine

        system.models_to_train[1]['coarse'] = nerf_coarse
        system.models_to_train[1]['fine'] = nerf_fine

        system.nerf_coarse = nerf_coarse
        system.nerf_fine = nerf_fine


        if hparams.continue_train_semantic:
            # Freeze weights for not semantic layers
            for param in system.parameters():
                param.requires_grad = False

            for param in system.nerf_coarse.semantic.parameters():
                param.requires_grad = True

            for param in system.nerf_fine.semantic.parameters():
                param.requires_grad = True


    trainer = Trainer(max_epochs=hparams.num_epochs,
                      max_steps=hparams.max_steps,
                      checkpoint_callback=checkpoint_callback,
                      val_check_interval = 1.0,
                      # resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=hparams.refresh_every,
                      gpus= hparams.num_gpus,
                      accelerator='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None)

    trainer.fit(system)
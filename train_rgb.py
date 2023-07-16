from config.train_config import get_opts
from train import run_train

if __name__ == '__main__':
    hparams = get_opts()
    run_train(hparams)
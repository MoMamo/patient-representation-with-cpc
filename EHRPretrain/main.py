from utils.yaml_act import yaml_load, yaml_save
from utils.arg_parse import arg_paser
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from model import *


def main():
    print('number of CUDA device available:', torch.cuda.device_count())

    args = arg_paser()

    # process config
    params = yaml_load(args.params)
    params.update(args.update_params)
    print(args)

    env_params, base_params, train_params,  callback_params = \
        params['env_params'], params['base_params'], \
        params['train_params'], params['callback_params']

    # set up logging and save updated config file
    save_path = args.params if args.save_path is None else args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    yaml_save(params, save_path + '/config.yaml')

    # define logger
    logger = TensorBoardLogger(save_path, name='my_log')

    print('initialize data loader')
    # create data loader
    input_fn = eval(base_params['dataloader'])
    train_params.update(base_params['dataloader_params'])

    dl = input_fn(data_path=train_params['data_path'], params=train_params)

    print('initialize model')
    # create the model and optimiser
    model = eval(base_params['model'])
    model_params = base_params['model_params']
    model_params.update(base_params['dataloader_params'])
    model = model(model_params)
    env_params.update({'logger': logger})
    env_params.update({'default_root_dir': os.path.join(save_path, 'checkpoint')})

    if args.accumulation is not None:
        env_params.update({'accumulate_grad_batches': int(args.accumulation)})

    model_params.update({'save_path': save_path})

    if train_params['mode'] == 'train':
        # set up checkpoint callbacks
        checkpoint_params = callback_params['modelCheckpoint']
        checkpoint_params.update({'dirpath': save_path})
        checkpoint_callback = ModelCheckpoint(**checkpoint_params)

        # set up additional callbacks
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [lr_monitor, checkpoint_callback]

        # add more callbacks if there is any
        plug_in = callback_params['plug_in']
        if plug_in is not None:
            for keys, value in plug_in.items():
                plugin_callback = eval(keys)
                callbacks.append(plugin_callback(**value))

        env_params.update({'callbacks': callbacks})
        trainer = pl.Trainer(**env_params)

        # train and evaluate model
        trainer.fit(model, dl)
    elif train_params['mode'] == 'eval':
        model.load_state_dict(torch.load(args.load_path, map_location=lambda storage, loc: storage)['state_dict'],
                              strict=False)

        trainer = pl.Trainer(**env_params)
        trainer.test(model, dl)
    else:
        raise ValueError()

if __name__ == "__main__":
    main()
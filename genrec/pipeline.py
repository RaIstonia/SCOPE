# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from typing import Union
import torch
import os
import json
from accelerate import Accelerator
from torch.utils.data import DataLoader

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer, LoadedTokenizer
from genrec.utils import get_config, init_seed, init_logger, init_device, \
    get_dataset, get_tokenizer, get_model, get_trainer, log
from genrec.utils import get_file_name

class Pipeline:
    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        checkpoint_path: str = None,
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )
        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device()
        self.checkpoint_path = checkpoint_path

        # Accelerator
        # Make tensorboard project_dir run-unique to avoid collisions when using W&B
        # (Accelerate uses project_name for both trackers; keeping project_dir unique is the robust fix.)
        run_slug = get_file_name(self.config, suffix='')
        self.project_dir = os.path.join(
            self.config['tensorboard_log_dir'],
            self.config["dataset"],
            self.config["model"],
            run_slug,
        )
        log_with = ['tensorboard']
        if bool(self.config.get('use_wandb', False)):
            log_with.append('wandb')
        self.accelerator = Accelerator(log_with=log_with, project_dir=self.project_dir)
        self.config['accelerator'] = self.accelerator

        # Seed and Logger
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'Device: {self.config["device"]}')

        # Dataset
        self.raw_dataset = get_dataset(dataset_name)(self.config)
        self.log(self.raw_dataset)
        self.split_datasets = self.raw_dataset.split()

        tokenized_dir = self.config.get('tokenized_dir')
        if tokenized_dir and os.path.isdir(tokenized_dir):
            from datasets import load_from_disk
            import json

            state_path = os.path.join(tokenized_dir, 'tokenizer_state.json')
            if not os.path.exists(state_path):
                self.log(f'[PIPELINE] tokenizer_state.json not found in {tokenized_dir}, fallback to normal tokenizer.', level='warning')
                if tokenizer is not None:
                    self.tokenizer = tokenizer(self.config, self.raw_dataset)
                else:
                    self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)
                self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)
            else:
                self.log(f'Loading tokenized data from {tokenized_dir} (skip tokenizer run)')
                self.tokenized_datasets = {
                    'train': load_from_disk(os.path.join(tokenized_dir, 'train')),
                    'val': load_from_disk(os.path.join(tokenized_dir, 'val')),
                    'test': load_from_disk(os.path.join(tokenized_dir, 'test')),
                }
                for split in self.tokenized_datasets:
                    self.tokenized_datasets[split].set_format(type='torch')
                with open(state_path, 'r') as f:
                    tokenizer_state = json.load(f)
                self.tokenizer = LoadedTokenizer(tokenizer_state, self.config, self.raw_dataset)
        else:
            if tokenizer is not None:
                self.tokenizer = tokenizer(self.config, self.raw_dataset)
            else:
                assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
                self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)
            self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)

        # Model
        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset, self.tokenizer)
            if checkpoint_path is not None:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.config['device']))
                self.log(f'Loaded model checkpoint from {checkpoint_path}')
        self.log(self.model)
        self.log(self.model.n_parameters)

        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:
            self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer)

    def run(self):
        # DataLoader
        train_dataloader = DataLoader(
            self.tokenized_datasets['train'],
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            collate_fn=self.tokenizer.collate_fn['train']
        )
        val_dataloader = DataLoader(
            self.tokenized_datasets['val'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['val']
        )
        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )

        if bool(self.config.get('record_test_curve', False)):
            best_epoch, best_val_score = self.trainer.fit(
                train_dataloader, val_dataloader, test_dataloader
            )
        else:
            best_epoch, best_val_score = self.trainer.fit(train_dataloader, val_dataloader)

        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)
        if self.checkpoint_path is None:
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        if self.accelerator.is_main_process and self.checkpoint_path is None:
            self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        use_graph = self.config.get('generate_w_decoding_graph', True)
        self.trainer.model.generate_w_decoding_graph = bool(use_graph)
        test_results = self.trainer.evaluate(test_dataloader)

        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})

        ckpt_for_test = self.checkpoint_path if self.checkpoint_path is not None else self.trainer.saved_model_ckpt
        self.log(f'Test ckpt={ckpt_for_test} | Test Results: {test_results}')

        self.trainer.end()
        return {
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'test_results': test_results,
        }

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)

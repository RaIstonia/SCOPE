# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
from logging import getLogger
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_scheduler

from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.evaluator import Evaluator
from genrec.utils import get_file_name, get_total_steps, config_for_log, log

class Trainer:
    """
    A class that handles the training process for a model.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the data.

    Attributes:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        evaluator (Evaluator): The evaluator used for evaluating the model.
        logger (Logger): The logger used for logging training progress.
        project_dir (str): The directory path for saving tensorboard logs.
        accelerator (Accelerator): The accelerator used for distributed training
        saved_model_ckpt (str): The file path for saving the trained model checkpoint.

    Methods:
        fit(train_dataloader, val_dataloader): Trains the model using the provided training and validation dataloaders.
        evaluate(dataloader, split='test'): Evaluate the model on the given dataloader.
        end(): Ends the training process and releases any used resources.
    """

    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='.pth')
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

    def fit(self, train_dataloader, val_dataloader, test_dataloader=None):

        self._curve_enabled = bool(self.config.get('record_test_curve', False)) \
            and test_dataloader is not None
        if self._curve_enabled:
            self._curve_test_dl = test_dataloader
            self._curve_interval = int(
                self.config.get('curve_eval_interval', self.config['eval_interval'])
            )
            self._curve_out_path = self.config.get('curve_out_path')
            self._curve_dataset = self.config.get('curve_dataset', self.config.get('dataset', '?'))
            self._curve_method = self.config.get('curve_method', self.config.get('model', '?'))
            default_spec = [{'curve': 'joint_full', 'ablate': [], 'score': None}]
            self._curve_specs = self.config.get('curve_specs', default_spec)
            if self._curve_out_path and self.accelerator.is_main_process:
                os.makedirs(os.path.dirname(os.path.abspath(self._curve_out_path)), exist_ok=True)
            self.log(f'[CURVE] per-epoch test curve recording ON → {self._curve_out_path} '
                     f'| interval={self._curve_interval} | specs={self._curve_specs}')

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return None, None

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_n_steps,
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        if self._curve_enabled:
            self._curve_test_dl = self.accelerator.prepare(self._curve_test_dl)

        if self.accelerator.is_main_process:
            try:
                cfg = config_for_log(self.config)
                self.log("[CONFIG] " + json.dumps(cfg, ensure_ascii=False, sort_keys=True))
            except Exception as e:
                self.log(f"[CONFIG] Failed to dump config: {e}", level="warning")

        # Trackers (tensorboard and optionally wandb)
        # - For W&B: use a stable project name and manage runs via group/name/tags.
        # - For TensorBoard: project_dir is already run-unique (set in Pipeline).
        project_name = get_file_name(self.config, suffix='')
        init_kwargs = {"tensorboard": {"flush_secs": 60}}
        if bool(self.config.get("use_wandb", False)):
            project_name = (self.config.get("wandb_project") or "SCOPE")
            wandb_kwargs = {}
            if self.config.get("wandb_entity"):
                wandb_kwargs["entity"] = self.config["wandb_entity"]
            if self.config.get("wandb_group"):
                wandb_kwargs["group"] = self.config["wandb_group"]
            if self.config.get("wandb_name"):
                wandb_kwargs["name"] = self.config["wandb_name"]
            else:
                wandb_kwargs["name"] = get_file_name(self.config, suffix="")
            if self.config.get("wandb_tags"):
                wandb_kwargs["tags"] = self.config["wandb_tags"]
            if self.config.get("wandb_mode"):
                wandb_kwargs["mode"] = self.config["wandb_mode"]
            init_kwargs["wandb"] = wandb_kwargs

        self.accelerator.init_trackers(
            project_name=project_name,
            config=config_for_log(self.config),
            init_kwargs=init_kwargs,
        )

        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            total_loss = 0.0

            total_loss_sem = 0.0
            total_loss_cf = 0.0
            total_loss_orth = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )
            for batch in train_progress_bar:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()

                if hasattr(outputs, "loss_sem"):
                    total_loss_sem += float(outputs.loss_sem.detach().item())
                if hasattr(outputs, "loss_cf"):
                    total_loss_cf += float(outputs.loss_cf.detach().item())
                if hasattr(outputs, "loss_orth"):
                    total_loss_orth += float(outputs.loss_orth.detach().item())

            avg_loss = total_loss / len(train_dataloader)
            log_dict = {"Loss/train_loss": avg_loss}
            msg = f'[Epoch {epoch + 1}] Train Loss: {avg_loss}'

            n_batches = len(train_dataloader)
            if total_loss_sem != 0.0 or total_loss_cf != 0.0 or total_loss_orth != 0.0:
                avg_sem = total_loss_sem / n_batches
                avg_cf = total_loss_cf / n_batches
                avg_orth = total_loss_orth / n_batches
                log_dict.update(
                    {
                        "Loss/train_loss_sem": avg_sem,
                        "Loss/train_loss_cf": avg_cf,
                        "Loss/train_loss_orth": avg_orth,
                    }
                )
                msg += f' (sem={avg_sem:.4f}, cf={avg_cf:.4f}, orth={avg_orth:.4f})'

            self.accelerator.log(log_dict, step=epoch + 1)
            self.log(msg)

            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                all_results = self.evaluate(val_dataloader, split='val')
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')
                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        if self.config['use_ddp']: # unwrap model for saving
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        self.log(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')

                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')

                    if self._curve_enabled:
                        self._record_test_curve(epoch + 1)
                    break

            if self._curve_enabled and (epoch + 1) % self._curve_interval == 0:
                self._record_test_curve(epoch + 1)
        self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')
        return best_epoch, best_val_score

    def _record_test_curve(self, epoch):

        model = self.accelerator.unwrap_model(self.model)
        has_survival = hasattr(model, 'set_survival_mode')

        prev_ablate = getattr(model, 'survival_ablate_streams', None)
        prev_score = getattr(model, 'survival_score_streams', None)
        prev_graph = getattr(model, 'generate_w_decoding_graph', None)

        was_training = model.training
        records = []
        try:
            for spec in self._curve_specs:
                ck = spec['curve']
                ablate = tuple(spec.get('ablate') or ())
                score = spec.get('score')
                if has_survival:
                    model.set_survival_mode(
                        ablate=ablate,
                        score=(tuple(score) if score else None),
                    )

                    if hasattr(model, 'refresh_mean_embeddings'):
                        model.refresh_mean_embeddings()
                elif ablate:

                    self.log(f'[CURVE] model has no set_survival_mode, skip curve={ck}',
                             level='warning')
                    continue

                model.generate_w_decoding_graph = False
                res = self.evaluate(self._curve_test_dl, split='test')
                if self.accelerator.is_main_process:
                    rec = {
                        'dataset': self._curve_dataset,
                        'method': self._curve_method,
                        'curve': ck,
                        'epoch': int(epoch),
                        'ndcg@10': float(res.get('ndcg@10', res.get('NDCG@10', float('nan')))),
                        'recall@10': float(res.get('recall@10', res.get('Recall@10', float('nan')))),
                    }
                    records.append(rec)
                    self.log(f'[CURVE] epoch={epoch} curve={ck} '
                             f"ndcg@10={rec['ndcg@10']:.4f}")
        finally:

            if has_survival:
                if prev_ablate is not None:
                    model.survival_ablate_streams = prev_ablate
                if prev_score is not None:
                    model.survival_score_streams = prev_score
                if hasattr(model, 'refresh_mean_embeddings'):
                    model.refresh_mean_embeddings()
            if prev_graph is not None:
                model.generate_w_decoding_graph = prev_graph
            if was_training:
                model.train()

        if records and self._curve_out_path and self.accelerator.is_main_process:
            with open(self._curve_out_path, 'a') as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']: # ddp, gather data from all devices for evaluation
                    preds = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    if isinstance(preds, tuple):
                        preds, n_visited_items = preds
                        all_preds, all_labels, all_n_visited_items = self.accelerator.gather_for_metrics((preds, batch['labels'], n_visited_items))
                        all_preds = (all_preds, all_n_visited_items)
                    else:
                        all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()
        output_results['n_visited_items'] = torch.cat(all_results['n_visited_items']).mean().item()
        return output_results

    def case_evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        diff2gap = defaultdict(list)

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                outputs = self.model.forward(batch, return_loss=False)
                states = outputs.final_states.gather(
                    dim=1,
                    index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.model.n_pred_head, self.model.config['n_embd'])
                )
                states = F.normalize(states, dim=-1)

                token_emb = self.model.gpt2.wte.weight[1:-1]
                token_emb = F.normalize(token_emb, dim=-1)
                token_embs = torch.chunk(token_emb, self.model.n_pred_head, dim=0)
                logits = [torch.matmul(states[:,0,i,:], token_embs[i].T) / self.model.temperature for i in range(self.model.n_pred_head)]
                logits = [F.log_softmax(logit, dim=-1) for logit in logits]
                token_logits = torch.cat(logits, dim=-1)    # (batch_size, n_tokens)

                sampled_items = torch.randint(1, self.model.item_id2tokens.shape[0], (token_logits.shape[0], 10))

                item_logits = torch.gather(
                    input=token_logits.unsqueeze(-2).expand(-1, sampled_items.shape[1], -1),              # (batch_size, n_items, n_tokens)
                    dim=-1,
                    index=(self.model.item_id2tokens[sampled_items,:] - 1)  # (batch_size, n_items, code_dim)
                ).mean(dim=-1)

                for batch_id in range(item_logits.shape[0]):
                    logit_list = item_logits[batch_id].cpu().tolist()
                    for i in range(len(logit_list)):
                        for j in range(i + 1, len(logit_list)):
                            item_a = sampled_items[batch_id, i]
                            item_b = sampled_items[batch_id, j]
                            gap = abs(logit_list[i] - logit_list[j])
                            diff = (self.model.item_id2tokens[item_a] != self.model.item_id2tokens[item_b]).sum().item()
                            diff2gap[diff].append(gap)
        return diff2gap

    def evaluate_cold_start(self, dataloader, token2item, item2group, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        group2results = {
            '0': defaultdict(list),
            '1': defaultdict(list),
            '2': defaultdict(list),
            '3': defaultdict(list),
            '4': defaultdict(list)
        }
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']: # ddp, gather data from all devices for evaluation
                    preds = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])

                for i, label in enumerate(batch['labels'].cpu().tolist()):
                    if self.config['model'] == 'TIGER':
                        item_id = token2item[' '.join(list(map(str, tuple(label[:-1]))))]
                    else:
                        item_id = token2item[str(label[0])]
                    if item_id not in item2group:
                        continue
                    group = item2group[item_id]
                    for key, value in results.items():
                        group2results[group][key].append(value.cpu().tolist()[i])

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()

        return output_results, group2results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)

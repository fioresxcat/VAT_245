import os

from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import ScriptModule
from torch.optim.optimizer import Optimizer
os.environ['TRANSFORMERS_CACHE'] = '/data/tungtx2/tmp/transformers_hub'

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from contextlib import contextmanager

import shutil
import json
from dataset import VATDataModule
from transformers import LiltForTokenClassification, LayoutLMv2FeatureExtractor, LayoutXLMTokenizerFast, LayoutXLMProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as lit
import pdb
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import torchmetrics
from torchmetrics import Metric
from collections import Counter, namedtuple
from pathlib import Path
from myutils import compute_accuracy

PredictionResult = namedtuple('PredictionResult', ['word', 'pred_idx', 'true_idx'])

class MyLiltForTokenClassification(nn.Module):
    def __init__(self, pretrained_path, label_list):
        super().__init__()
        self.LABEL_LIST = label_list
        self.LABEL2ID = {lb: id for id, lb in enumerate(label_list)}
        self.ID2LABEL = {v: k for k, v in self.LABEL2ID.items()}
        self.lilt = LiltForTokenClassification.from_pretrained(pretrained_path, id2label=self.ID2LABEL)
    
    def forward(self, input_ids, bbox, attention_mask):
        outputs = self.lilt(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, return_dict=False)
        logits = outputs[0]
        return logits
    
    

class LiltClassifierModule(lit.LightningModule):
    def __init__(
        self,
        label_list: list,
        model: MyLiltForTokenClassification,
        learning_rate: float,
        reset_optimizer: bool,
        n_warmup_epochs: int,
        class_weight: list,
        save_pred: bool,
        save_pred_dir: str,
    ):
        super().__init__()
        self.LABEL_LIST = label_list
        self.LABEL2ID = {lb: id for id, lb in enumerate(label_list)}
        self.ID2LABEL = {v: k for k, v in self.LABEL2ID.items()}
        self.model = model
        self.num_classes = len(label_list)
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.reset_optimizer = reset_optimizer
        self.n_warmup_epochs = n_warmup_epochs
        self.save_pred = save_pred
        self.save_pred_dir = save_pred_dir
        self._init_loss_and_metrics()
        

    def _init_loss_and_metrics(self):
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,
            weight=torch.tensor(self.class_weight, dtype=torch.float)
        )

        self.predict_pred, self.predict_true = torch.tensor([], dtype=torch.int, device=self.device), torch.tensor([], dtype=torch.int, device=self.device)

        self.micro_train_acc = MulticlassAccuracy(num_classes=self.num_classes, average='micro', ignore_index=-100)
        self.micro_val_acc = MulticlassAccuracy(num_classes=self.num_classes, average='micro', ignore_index=-100)
        self.micro_test_acc = MulticlassAccuracy(num_classes=self.num_classes, average='micro', ignore_index=-100)

        self.macro_train_acc = MulticlassAccuracy(num_classes=self.num_classes, average='macro', ignore_index=-100)
        self.macro_val_acc = MulticlassAccuracy(num_classes=self.num_classes, average='macro', ignore_index=-100)
        self.macro_test_acc = MulticlassAccuracy(num_classes=self.num_classes, average='macro', ignore_index=-100)



    def _compute_loss_and_outputs(self, input_ids, bbox, attention_mask, labels):
        logits = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
        )   # shape n x 512 x 19
        loss = self.criterion(logits.view(-1, self.num_classes), labels.view(-1))
        return loss, logits
    
    
    def step(self, batch, batch_idx, split):
        input_ids, bbox, attention_mask, labels = batch['input_ids'], batch['bbox'], batch['attention_mask'], batch['labels']
        loss, logits = self._compute_loss_and_outputs(input_ids, bbox, attention_mask, labels)
        pred = torch.argmax(logits.view(-1, self.num_classes), dim=-1)

        macro_acc = getattr(self, f'macro_{split}_acc')
        macro_acc(pred, labels.view(-1))

        micro_acc = getattr(self, f'micro_{split}_acc')
        micro_acc(pred, labels.view(-1))

        self.log_dict({
            f'{split}_loss': loss,
            f'balanced_{split}_acc': macro_acc,
            f'micro_{split}_acc': micro_acc
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss


    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='val')
        
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='test')


    def predict_step(self, batch, batch_idx):
        encoded_inputs, word_ids, anno_info = batch
        input_ids, bbox, att_mask, labels = encoded_inputs['input_ids'], encoded_inputs['bbox'], encoded_inputs['attention_mask'], encoded_inputs['labels']
        img_fp, json_fp, words, orig_polys, normalized_boxes, text_labels = list(anno_info.values())
        orig_polys = tuple([tuple([tuple(pt) for pt in poly]) for poly in orig_polys])

        # infer
        logits = self.model(input_ids, bbox, att_mask)    # shape n x 512 x 19
        logits = logits.cpu().numpy()

        # process res
        wordidx2pred_final = get_final_pred(logits, word_ids)
        res = {}
        preds, trues = [], []
        for word_idx, pred_idx in wordidx2pred_final.items():
            real_text_label = text_labels[word_idx]
            true_idx = self.LABEL2ID[real_text_label]
            preds.append(pred_idx)
            trues.append(true_idx)
            res[tuple(orig_polys[word_idx])] = PredictionResult(word=words[word_idx], pred_idx=pred_idx, true_idx=true_idx)

        self.predict_pred = torch.concat([self.predict_pred, torch.tensor(preds, device=self.predict_pred.device)])
        self.predict_true = torch.concat([self.predict_true, torch.tensor(trues, device=self.predict_true.device)])

        # compute acc
        correct = 0
        total = 0
        for i in range(len(trues)):
            if trues[i] == -100:
                continue
            if preds[i] == trues[i]:
                correct += 1
            else:
                print(f'{json_fp} wrong: word: {words[i]}, true_label: {self.ID2LABEL[trues[i]]}, pred_label: {self.ID2LABEL[preds[i]]}')
            total += 1
        acc = correct / total
        print('acc: ', acc)
            
        
        if self.save_pred:
            os.makedirs(self.save_pred_dir, exist_ok=True)
            json_data = json.load(open(json_fp))
            ls_predicted_polys = list(res.keys())
            for i, shape in enumerate(json_data['shapes']):
                # json_data['shapes'][i]['label'] = 'text'
                poly = tuple([tuple(pt) for pt in shape['points']])
                if poly in ls_predicted_polys:
                    json_data['shapes'][i]['label'] = self.ID2LABEL[res[poly].pred_idx]
            
            save_pred_dir = os.path.join(self.save_pred_dir, Path(json_fp).parent.parent.name, Path(json_fp).parent.name)
            os.makedirs(save_pred_dir, exist_ok=True)
            # save_pred_dir = self.save_pred_dir
            with open(os.path.join(save_pred_dir, Path(json_fp).name), 'w') as f:
                json.dump(json_data, f)
            shutil.copy(img_fp, save_pred_dir)


    
    def on_predict_epoch_end(self, results: List[Any]) -> None:
        macro_acc = torchmetrics.functional.accuracy(
                self.predict_pred, 
                self.predict_true, 
                task='multiclass', 
                num_classes=self.num_classes, 
                average='macro', 
                ignore_index=-100
            ) * self.num_classes / (torch.unique(self.predict_true).size(0))
        micro_acc = torchmetrics.functional.accuracy(
            self.predict_pred, 
            self.predict_true, 
            task='multiclass', 
            num_classes=self.num_classes, 
            average='micro', 
            ignore_index=-100
        )
        print('macro acc: ', macro_acc)
        print('micro acc: ', micro_acc)


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.trainer.callbacks[0].mode,
            factor=0.2,
            patience=8,
        )

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     opt,
        #     gamma=0.97,
        # )

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.trainer.callbacks[0].monitor,
                'frequency': 1,
                'interval': 'epoch'
            }
        }


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_lbfgs=False):
        if self.trainer.current_epoch <= self.n_warmup_epochs:
            lr_scale = 0.75 ** (self.n_warmup_epochs - self.trainer.current_epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.learning_rate

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    
    def on_fit_start(self) -> None:
        if self.reset_optimizer:
            opt = type(self.trainers.optimizers[0])(self.parameters(), **self.trainer.optimizers[0].defaults)
            self.trainer.optimizers[0].load_state_dict(opt.state_dict())
            print('Optimizer reseted')
        


    def on_train_epoch_end(self) -> None:
        print('\n')

    
    def to_onnx(self, save_path: str, opset=14):
        self.eval()
        input_ids = torch.randint(low=0, high=10000, size=(2, 512))
        bbox = torch.randint(low=0, high=1000, size=(2, 512, 4))
        att_mask = torch.randint(low=0, high=2, size=(2, 512))

        torch.onnx.export(
            self.model,
            ({
                'bbox': bbox,
                'attention_mask': att_mask,
                'input_ids': input_ids
            }),
            save_path,
            input_names=['input_ids', 'bbox', 'attention_mask'],
            output_names = ['output'],
            dynamic_axes = {
                "input_ids": {0: "batch_size"},
                "bbox": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            opset_version=opset
        )
        print(f'Onnx model savevd to {save_path}')
    

    def to_torchscript(self, file_path: str, method: str = "script", example_inputs = None, **kwargs) :
        mode = self.training

        if method == "script":
            with _jit_is_scripting():
                torchscript_module = torch.jit.script(self.model.eval(), **kwargs)
        elif method == "trace":
            if example_inputs is None:
                if self.example_input_array is None:
                    raise ValueError(
                        "Choosing method=`trace` requires either `example_inputs`"
                        " or `model.example_input_array` to be defined."
                    )
                example_inputs = self.example_input_array

            # automatically send example inputs to the right device and use trace
            example_inputs = self._on_before_batch_transfer(example_inputs)
            example_inputs = self._apply_batch_transfer_handler(example_inputs)
            with _jit_is_scripting():
                torchscript_module = torch.jit.trace(func=self.model.eval(), example_inputs=example_inputs, **kwargs)
        else:
            raise ValueError(f"The 'method' parameter only supports 'script' or 'trace', but value given was: {method}")

        self.train(mode)

        if file_path is not None:
            with open(file_path, "wb") as f:
                torch.jit.save(torchscript_module, f)

        return torchscript_module


@contextmanager
def _jit_is_scripting():
    """Workaround for https://github.com/pytorch/pytorch/issues/67146."""
    lit.LightningModule._jit_is_scripting = True
    try:
        yield
    finally:
        lit.LightningModule._jit_is_scripting = False


def get_final_pred(logits, word_ids):
    wordidx2pred = {}
    for idx in range(len(logits)):
        output = logits[idx]
        preds_val = output.tolist()
        word_indices = word_ids[idx]
        for i, (pred, word_idx) in enumerate(zip(preds_val, word_indices)):
            if word_idx is None:
                continue
            if word_idx not in wordidx2pred:
                wordidx2pred[word_idx] = [(np.argmax(pred), np.max(pred))]
            else:
                wordidx2pred[word_idx].append((np.argmax(pred), np.max(pred)))

    wordidx2pred_final = {}
    for word_idx, pred in wordidx2pred.items():
        ls_pred_idx = [el[0] for el in pred]
        ls_pred_scores = [el[1] for el in pred]
        if len(ls_pred_idx) == 2:
            max_score_idx = np.argmax(ls_pred_scores)
            final_pred = ls_pred_idx[max_score_idx]
        else:
            final_pred = Counter(ls_pred_idx).most_common(1)[0][0]
            
        wordidx2pred_final[word_idx] = final_pred
    
    return wordidx2pred_final



import torch
import torchmetrics
import numpy as np
import onnx
import onnxruntime
from lmv3 import LMv3ClassifierModule, get_final_pred, PredictionResult, BalancedAccuracy
from tqdm import tqdm
from dataset import VATDataModule


class LMv3Onnx:
    def __init__(self, onnx_path):
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        assert onnxruntime.get_device() == 'GPU', 'onnx not running on GPU!'

        print('------------- ONNX model summary ------------')
        for input in self.ort_session.get_inputs():
            print(input.name, '-', input.type, '-', input.shape)
        print()
    

    def forward(self, input_ids, bbox, att_mask, pixel_values):
        outputs = self.ort_session.run(
            None,
            {
                'input_ids': input_ids.astype(np.int64),
                'bbox': bbox.astype(np.int64),
                'attention_mask': att_mask.astype(np.int64),
                'pixel_values': pixel_values
            }
        )
        return outputs[0]
    

    def predict(self, ds_module: VATDataModule, save_res_dir=None, stage='predict'):
        ds_module.setup(stage=stage)
        micro_acc = torchmetrics.Accuracy(task='multiclass', num_classes=len(ds_module.LABEL_LIST), average='micro')
        balanced_acc = BalancedAccuracy(num_classes=len(ds_module.LABEL_LIST))
        all_preds, all_trues = [], []
        if save_res_dir is not None:
            os.makedirs(save_res_dir, exist_ok=True)

        for item in tqdm(ds_module.predict_dataloader()):
            encoded_inputs, word_ids, anno_info = item
            img_fp, json_fp, words, orig_polys, boxes, text_labels = list(anno_info.values())
            orig_polys = tuple([tuple([tuple(pt) for pt in poly]) for poly in orig_polys])

            # infer
            logits = self.forward(
                encoded_inputs['input_ids'].numpy(),
                encoded_inputs['bbox'].numpy(),
                encoded_inputs['attention_mask'].numpy(),
                encoded_inputs['pixel_values'].numpy(),
            )   # shape n x 512 x 19

            # process res
            wordidx2pred_final = get_final_pred(logits, word_ids)
            res = {}
            preds, trues = [], []
            for word_idx, pred_idx in wordidx2pred_final.items():
                real_text_label = text_labels[word_idx]
                true_idx = ds_module.LABEL2ID[real_text_label]
                preds.append(pred_idx)
                trues.append(true_idx)
                res[tuple(orig_polys[word_idx])] = PredictionResult(word=words[word_idx], pred_idx=pred_idx, true_idx=true_idx)
            all_preds.extend(preds)
            all_trues.extend(trues)
            
            if save_res_dir is not None:
                os.makedirs(save_res_dir, exist_ok=True)
                json_data = json.load(open(json_fp))
                ls_predicted_polys = list(res.keys())
                for i, shape in enumerate(json_data['shapes']):
                    poly = tuple([tuple(pt) for pt in shape['points']])
                    if poly in ls_predicted_polys:
                        json_data['shapes'][i]['label'] = ds_module.ID2LABEL[res[poly].pred_idx]
                
                with open(os.path.join(save_res_dir, Path(json_fp).name), 'w') as f:
                    json.dump(json_data, f)
                shutil.copy(img_fp, save_res_dir)


        print('Micro acc: ', micro_acc(torch.tensor(all_preds), torch.tensor(all_trues)))
        print('Balanced acc: ', balanced_acc(torch.tensor(all_preds), torch.tensor(all_trues)))


    


if __name__ == '__main__':
    import pdb
    from dataset import VATDataModule
    from pathlib import Path
    import os
    import shutil
    import json
    import yaml
    from easydict import EasyDict

    onnx_path = 'ckpt/exp11-lmv3classifier-fp32/epoch30.onnx'
    predictor = LMv3Onnx(onnx_path)

    # get config file in the onnx dir
    with open(os.path.join(Path(onnx_path).parent, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    ds_module = VATDataModule(
        predict_dirs=['VAT_ie_data/VAT_ie_data_old/test_VNG_pdf_2_extracted'],
        label_list=config.data.label_list,
        processor_path='microsoft/layoutlmv3-base',
        remove_accent=True,
        keep_pixel_values=True,
        ls_disable_label=[],
        augment=False,
        stride=128,
        carefully_choose_idx=False,
        batch_size=4,
        num_workers=0,
    )

    predictor.predict(ds_module, save_res_dir='results/exp11-ep30/test_VNG_pdf_2_extracted')
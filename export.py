import torch
from lmv3 import LMv3ClassifierModule, LMv3ClassifierModel, Lmv3ForTokenClassification
import yaml
from easydict import EasyDict
from pathlib import Path
import os

if __name__ == '__main__':
    ckpt_path = 'ckpt/lmv3/exp21_new_ocr/epoch=43-train_loss=0.327-balanced_train_f1=0.998-micro_train_f1=1.000-val_loss=0.331-balanced_val_f1=0.994-micro_val_f1=0.999.ckpt'
    with open(os.path.join(Path(ckpt_path).parent, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)    

    # model = LMv3ClassifierModel(
    #     pretrained_path=config.model.model.init_args.pretrained_path,
    #     label_list=config.model.model.init_args.label_list,
    #     hidden_dropout=config.model.model.init_args.hidden_dropout,
    #     cls_dropout=config.model.model.init_args.cls_dropout
    # )

    model = Lmv3ForTokenClassification(
        label_list=config.data.label_list,
        pretrained_path=config.model.model.init_args.pretrained_path
    )

    lmv3_module = LMv3ClassifierModule.load_from_checkpoint(
        ckpt_path,
        label_list=config.data.label_list,
        model=model,
        learning_rate=config.model.label_list,
        reset_optimizer=False,
        n_warmup_epochs=0,
        class_weight = config.model.class_weight,
        save_pred=False,
        save_pred_dir=False
    )
    lmv3_module.to_onnx('ckpt/lmv3/exp21_new_ocr/epoch43.onnx', opset=14)

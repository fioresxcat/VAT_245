import yaml
from easydict import EasyDict


DEFAULT_MODEL_PATH = {
    'layoutlmv3': 'microsoft/layoutlmv3-base',
    'lilt': 'SCUT-DLVCLab/lilt-infoxlm-base'
}



class Config:
    def __init__(self):
        self.data = EasyDict(dict(
            train_dir = "/data/tungtx2/huggingface/latest_data_245_final_2/train",
            val_dir = "/data/tungtx2/huggingface/latest_data_245_final_2/test",
            ls_exclude_dir = []
        ))

        self.model = EasyDict(dict(
            pretrained_path = None,
            type = "layoutlmv3"
        ))

        
        self.training = EasyDict(dict(
            output_dir = "ckpt/unified/last_model_final/lmv3_train+val",
            num_train_epochs = 100,
            learning_rate = 5e-5,
            weight_decay = 0.01,
            metric_for_best_model = 'f1',
            greater_is_better = True,
            warmup_ratio = 0.1,
            ls_disable_marker = [],
            augment = True,
            remove_accent = True,
            stride = 128,
            carefully_choose_idx = True,
            batch_size = 4
        ))

        self._post_init()
    
    def _post_init(self):
        if self.model.pretrained_path is None:
            self.model.pretrained_path = DEFAULT_MODEL_PATH[self.model.type]

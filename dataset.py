from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from myutils import *
import numpy as np
import lightning.pytorch as lit
from typing import List, Tuple, Union
from collections import OrderedDict
from typing_extensions import Literal
import transformers
    

class VATDataset(Dataset):
    def __init__(
        self,
        data_dirs: List[str],
        mode: Literal["fit", "validate", "test", "predict"],
        processor,
        label2id: dict,
        keep_pixel_values: bool = True,
        ls_disable_label: List = [], 
        remove_accent: bool = False,
        stride: int = 128,
        carefully_choose_idx: bool = False,
        augment: bool = False,
        augment_props: dict = {},
        normal_transforms=None,
        geometric_transforms=None
    ):
        
        self.mode = mode
        self.ls_img_fp, self.ls_json_fp = [], []
        for data_dir in data_dirs:
            ls_img_fp, ls_json_fp = get_img_and_json_file_paths(data_dir)
            self.ls_img_fp.extend(ls_img_fp)
            self.ls_json_fp.extend(ls_json_fp)

        assert len(self.ls_img_fp) == len(self.ls_json_fp)

        self.processor = processor
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.keep_pixel_values = keep_pixel_values
        self.ls_disable_label = ls_disable_label
        self.augment = augment
        self.augment_props = augment_props
        self.normal_transforms = normal_transforms
        self.geometric_transforms = geometric_transforms
        self.remove_accent = remove_accent
        self.stride = stride
        self.carefully_choose_idx = carefully_choose_idx
        
        if self.carefully_choose_idx:
            self.multi_split = {}


    def __len__(self):
        return len(self.ls_img_fp)


    def __getitem__(self, index):
        img_fp = self.ls_img_fp[index]
        json_fp = self.ls_json_fp[index]
        img, words, orig_polys, normalized_boxes, text_labels = gen_annotation_for_img(
            self.mode,
            img_fp, 
            json_fp, 
            ls_disable_label=self.ls_disable_label, 
            remove_accent=self.remove_accent,
            augment=self.augment if self.mode=='train' else False,
            normal_transforms=self.normal_transforms,
            geometric_transforms=self.geometric_transforms,
            **self.augment_props
        )
        idx_labels = [self.label2id[label] for label in text_labels]

        encoded_inputs = self.processor(
            img, 
            words, 
            boxes=normalized_boxes, 
            word_labels=idx_labels, 
            truncation=True, 
            stride=self.stride, 
            padding="max_length", 
            max_length=512, 
            return_overflowing_tokens=True, 
            return_offsets_mapping=True, 
            return_tensors="pt"
        )
        encoded_inputs.pop('overflow_to_sample_mapping')
        encoded_inputs.pop('offset_mapping')
        if not self.keep_pixel_values:
            encoded_inputs.pop('image')

        if self.mode == 'train':
            # remove batch dimension    
            if self.carefully_choose_idx:
                if str(img_fp) not in self.multi_split:
                    idx = np.random.randint(0, len(encoded_inputs['bbox']))
                    self.multi_split[str(img_fp)] = [idx]
                else:
                    if len(self.multi_split[str(img_fp)]) >= len(encoded_inputs['bbox']):
                        idx = np.random.randint(0, len(encoded_inputs['bbox']))
                    else:
                        ls_available_idx = list(set(range(len(encoded_inputs['bbox']))) - set(self.multi_split[str(img_fp)]))
                        idx = np.random.choice(ls_available_idx)
                        self.multi_split[str(img_fp)].append(idx)
            else:
                idx = np.random.randint(0, len(encoded_inputs['bbox']))

            for k, v in encoded_inputs.items():
                encoded_inputs[k] = v[idx]

        elif self.mode in ['validate', 'test', 'predict'] and self.keep_pixel_values:
                encoded_inputs['pixel_values'] = torch.stack(encoded_inputs['pixel_values'], dim=0)
        
        if self.mode != 'predict':
            return encoded_inputs
        else:
            anno_info = OrderedDict(
                img_fp=img_fp,
                json_fp=json_fp,
                words=words,
                orig_polys=orig_polys,
                boxes=normalized_boxes,
                text_labels=text_labels
            )
            word_ids = [encoded_inputs.word_ids(i) for i in range(encoded_inputs['bbox'].shape[0])]
            return encoded_inputs, word_ids, anno_info




class VATDataModule(lit.LightningDataModule):
    def __init__(
        self,
        label_list: List,
        batch_size: int,
        num_workers: int,
        processor_path: str,
        remove_accent: bool,     
        keep_pixel_values: bool,
        ls_disable_label: List,
        augment: bool,
        augment_props: dict = {},
        train_dirs: List[str] = [],
        val_dirs: List[str] = [], 
        test_dirs: List[str] = [],
        predict_dirs: List[str] = [],
        stride: int = 128,
        carefully_choose_idx: bool = False,
        
    ):
        super().__init__()
        self.LABEL_LIST = label_list
        self.LABEL2ID = {lb: id for id, lb in enumerate(label_list)}
        self.ID2LABEL = {v: k for k, v in self.LABEL2ID.items()}

        # check data health first
        for data_dir in train_dirs + val_dirs + test_dirs + predict_dirs:
            if data_dir is None:
                continue
            is_ok, invalid_points, invalid_labels, invalid_texts = self.check_data_health(data_dir, self.LABEL_LIST, ls_disable_label, del_invalid_box=False)
            if is_ok:
                print(f'{data_dir} PASSED')
            else:
                print(f'{data_dir} WRONG')
                pdb.set_trace()

        # dataset related
        self.train_dirs = train_dirs
        self.val_dirs = val_dirs
        self.test_dirs = test_dirs
        self.predict_dirs = predict_dirs
        self.processor = get_processor_from_path(processor_path)
        self.augment = augment
        self.augment_props = augment_props
        self.common_data_args = dict(
            label2id=self.LABEL2ID,
            keep_pixel_values=keep_pixel_values,
            ls_disable_label=ls_disable_label,
            remove_accent=remove_accent,
            stride=stride,
            carefully_choose_idx=carefully_choose_idx,
            processor=self.processor
        )

        # training related
        self.batch_size = batch_size
        self.num_workers = num_workers


    def prepare_data(self) -> None:
        return super().prepare_data()


    def setup(self, stage=None):
        if stage in ['fit', 'validate']:
            self._init_transforms()
            self.train_ds = VATDataset(
                data_dirs=self.train_dirs,
                mode='train',
                augment=self.augment,
                augment_props=self.augment_props,
                normal_transforms=self.normal_transforms,
                geometric_transforms=self.geometric_transforms,
                **self.common_data_args
            )
            self.val_ds = VATDataset(
                data_dirs=self.val_dirs,
                mode='validate',
                augment=False,
                **self.common_data_args
            )
        elif stage == 'test':
            self.test_ds = VATDataset(
                data_dirs=self.test_dirs,
                mode='test',
                augment=False,
                **self.common_data_args
            )
        elif stage == 'predict':
            self.predict_ds = VATDataset(
                data_dirs=self.predict_dirs,
                mode='predict',
                augment=False,
                **self.common_data_args
            )
    

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_ds, batch_size=None, num_workers=self.num_workers, shuffle=False)


    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=None, num_workers=self.num_workers, shuffle=False)


    def predict_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.predict_ds, batch_size=None, num_workers=self.num_workers, shuffle=False)    


    def _init_transforms(self):
        self.normal_transforms = A.Compose([
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3,5), p=0.3),
        ])

        self.geometric_transforms = A.Compose([
            A.OneOf([
                A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
                # A.Affine(p=0.3, 
                #     scale=1, 
                #     translate_percent={
                #         'x': (0, 0.1),
                #         'y': (0, 0.1)
                #     }, 
                #     rotate=0, 
                #     shear={
                #         'x': (-7, 7),
                #         'y': (-7, 7)
                #     }, 
                #     mode=cv2.BORDER_CONSTANT, 
                #     cval=(255, 255, 255), 
                #     fit_output=False
                # ),
                A.SafeRotate(p=0.5, limit=7, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            ])
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


    @staticmethod
    def check_data_health(data_dir, label_list, ls_disable_label, del_invalid_box=False):
        ls_json_fp = sorted(list(Path(data_dir).rglob('*.json')))
        invalid_points, invalid_labels, invalid_texts = [], [], []
        for fp in ls_json_fp:
            json_data = json.load(open(fp))
            invalid_indices = []
            for i, shape in enumerate(json_data['shapes']):
                if len(shape['points']) != 4:
                    print(f'{fp} contains shape with {len(shape["points"])} points!')
                    invalid_points.append(str(fp))
                    invalid_indices.append(i)

                if shape['label'] not in label_list and shape['label'] not in ls_disable_label:
                    print(f'{fp} contains shape with label {shape["label"]} not in label_list!')
                    invalid_labels.append(str(fp))
                    
                if 'text' not in shape:
                    print(f'{fp} contains shape without text!')
                    invalid_texts.append(str(fp))

            if len(invalid_indices) > 0 and del_invalid_box:
                json_data['shapes'] = [shape for i, shape in enumerate(json_data['shapes']) if i not in invalid_indices]
                json.dump(json_data, open(fp, 'w'))

        is_ok = True if len(invalid_labels+invalid_points+invalid_texts) == 0 else False
        return is_ok, invalid_points, invalid_labels, invalid_texts



if __name__ == '__main__':
    import pdb

    ds_module = VATDataModule(
        train_dirs='VAT_ie_data/VAT_ie_data_old/train',
        val_dirs='VAT_ie_data/VAT_ie_data_old/val',
        test_dirs='VAT_ie_data/VAT_ie_data_old/test',
        predict_dirs='VAT_ie_data/VAT_ie_data_old/test',
        processor_path='microsoft/layoutlmv3-base'
    )
    ds_module.setup(stage='predict')

    for item in ds_module.predict_ds:
        encoded_inputs = item
        pdb.set_trace()

    

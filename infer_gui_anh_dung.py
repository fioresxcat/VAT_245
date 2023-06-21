import os
import json
import cv2
from PIL import Image
import numpy as np
import unidecode
from transformers import LayoutLMv3Processor
import torch
import onnx, onnxruntime
from collections import Counter

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


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


def gen_annotation_for_img(
    img_fp, 
    json_fp, 
    ls_disable_label=[], 
    remove_accent=True, 
):
    
    img = Image.open(img_fp).convert("RGB")
    json_data = json.load(open(json_fp))

    words, orig_polys, boxes, labels = [], [], [], []
    img_w, img_h = img.size
    for i, shape in enumerate(json_data['shapes']):
        label = shape['label'] if shape['label'] not in ls_disable_label else 'text'
        words.append(unidecode.unidecode(shape['text'].lower()))
        labels.append(label)
        pts = [coord for pt in shape['points'] for coord in pt]
        xmin = np.clip(min(pts[0::2]), 0, img_w)
        xmax = np.clip(max(pts[0::2]), 0, img_w)
        ymin = np.clip(min(pts[1::2]), 0, img_h)
        ymax = np.clip(max(pts[1::2]), 0, img_h)
        boxes.append((xmin, ymin, xmax, ymax))
        orig_polys.append(tuple([tuple(pt) for pt in shape['points']]))

    normalized_boxes = [normalize_bbox(box, img_w, img_h) for box in boxes]
    return img, words, orig_polys, normalized_boxes, labels



if __name__ == '__main__':
        onnx_path = ''
        img_fp, json_fp = '', ''
        label_list = [
            'text',
            'doc_type',
            'doi',
            'sign',
            'invoice_number',
            'provider_tax',
            'provider_name',
            'provider_add',
            'provider_phone',
            'provider_bank',
            'provider_bank_acc',
            'provider_web',
            'customer_name',
            'customer_tax',
            'vat_amount',
            'total_amount',
            'sign_customer_name',
            'sign_date'
        ]
        label2id = {label: id for id, label in enumerate(label_list)}

        # model
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        assert onnxruntime.get_device() == 'GPU', 'onnx not running on GPU!'

        print('------------- ONNX model summary ------------')
        for input in ort_session.get_inputs():
            print(input.name, '-', input.type, '-', input.shape)
        print()

        # processor
        processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)
        processor.tokenizer.only_label_first_subword = False


        img, words, orig_polys, normalized_boxes, text_labels = gen_annotation_for_img(img_fp, json_fp, ls_disable_label=[], remove_accent=True)
        idx_labels = [label2id[label] for label in text_labels]
        encoded_inputs = processor(
            img, 
            words, 
            boxes=normalized_boxes, 
            word_labels=idx_labels, 
            truncation=True, 
            stride=128, 
            padding="max_length", 
            max_length=512, 
            return_overflowing_tokens=True, 
            return_offsets_mapping=True, 
            return_tensors="np"
        )
        encoded_inputs.pop('overflow_to_sample_mapping')
        encoded_inputs.pop('offset_mapping')
        encoded_inputs['pixel_values'] = torch.stack(encoded_inputs['pixel_values'], dim=0)


        outputs = ort_session.run(
            None,
            {
                'input_ids': encoded_inputs['input_ids'].astype(np.int64),
                'bbox': encoded_inputs['bbox'].astype(np.int64),
                'attention_mask': encoded_inputs['att_mask'].astype(np.int64),
                'pixel_values': encoded_inputs['pixel_values'].astype(np.float32)
            }
        )

        orig_polys = tuple([tuple([tuple(pt) for pt in poly]) for poly in orig_polys])

        # infer
        logits = ort_session.run(
            encoded_inputs['input_ids'].numpy(),
            encoded_inputs['bbox'].numpy(),
            encoded_inputs['attention_mask'].numpy(),
            encoded_inputs['pixel_values'].numpy(),
        )   # shape n x 512 x 19

        # process res
        word_ids = [encoded_inputs.word_ids(i) for i in range(encoded_inputs['bbox'].shape[0])]
        wordidx2pred_final = get_final_pred(logits, word_ids)
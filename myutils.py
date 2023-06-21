import os
import json
from pathlib import Path
import numpy as np
import yaml
import torch
import unidecode
from PIL import Image, ImageDraw, ImageFont
import pdb
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import cv2
import albumentations as A



def compute_accuracy(pred, true, ignore_idx=-100):
    correct = 0
    total = 0
    for i in range(len(true)):
        if true[i] == ignore_idx:
            continue
        if pred[i] == true[i]:
            correct += 1
        total += 1
    accuracy = correct / total
    return accuracy


def get_img_fp_from_json_fp(json_fp):
    if isinstance(json_fp, str):
        json_fp = Path(json_fp)
    ls_ext = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    for ext in ls_ext:
        img_fp = json_fp.with_suffix(ext)
        if img_fp.exists():
            return img_fp
    return None


def is_in_dirs(fp, dirs):
    is_in = False
    for directory in dirs:
        if str(directory) in str(fp):
            return True
    return False



def get_img_and_json_file_paths(data_dir, exclude_dirs=[]):
    ls_img_fp, ls_json_fp = [], []
    for json_fp in Path(data_dir).rglob('*.json'):
        if is_in_dirs(json_fp, exclude_dirs):
            continue

        img_fp = get_img_fp_from_json_fp(json_fp)
        ls_img_fp.append(str(img_fp))
        ls_json_fp.append(str(json_fp))
    
    return ls_img_fp, ls_json_fp



def find_all_labels(data_dir, exclude_dirs=[]):
    labels = []
    for jp in Path(data_dir).rglob('*.json'):
        if is_in_dirs(jp, exclude_dirs):
            continue

        data = json.load(open(jp))
        labels.extend([shape['label'] for shape in data['shapes']])

    return set(labels)

from collections import Counter
def count_all_labels(data_dir, exclude_dirs=[]):
    labels = []
    for jp in Path(data_dir).rglob('*.json'):
        if is_in_dirs(jp, exclude_dirs):
            continue

        data = json.load(open(jp))
        labels.extend([shape['label'] for shape in data['shapes']])

    return Counter(labels)



def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

    

def get_processor_from_path(processor_path):
    from transformers import AutoProcessor, LayoutLMv3Processor, LayoutLMv2FeatureExtractor, LayoutXLMTokenizerFast, LayoutXLMProcessor


    if 'layoutlmv3' in processor_path:
        processor = LayoutLMv3Processor.from_pretrained(processor_path, apply_ocr=False)
        processor.tokenizer.only_label_first_subword = False

    elif 'lilt' in processor_path:
        feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(processor_path)
        tokenizer.only_label_first_subword = False
        processor = LayoutXLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    else:
        raise ValueError('processor type not supported!')

    return processor


def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    objs = root.findall('object')
    boxes, obj_names = [], []
    for obj in objs:
        obj_name = obj.find('name').text
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        obj_names.append(obj_name)
    return boxes, obj_names


def widen_box(box, percent_x, percent_y):
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        h = ymax - ymin
        xmin -= w * percent_x
        ymin -= h * percent_y
        xmax += w * percent_x
        ymax += h * percent_y
        return (int(xmin), int(ymin), int(xmax), int(ymax))

    
def draw_json_on_img(img, json_data):
    labels = list(set(shape['label'] for shape in json_data['shapes']))
    color = {}
    for i in range(len(labels)):
        color[labels[i]] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        
    img = img.copy()
    draw = ImageDraw.Draw(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5# Draw the text on the image
    # font = ImageFont.truetype(font.font.family, font_size)
    for i, shape in enumerate(json_data['shapes']):
        polys = shape['points']
        polys = [(int(pt[0]), int(pt[1])) for pt in polys]
        label = shape['label']
        draw.polygon(polys, outline=color[label], width=2)
        # Draw the text on the image
        img = np.array(img)
        cv2.putText(img, shape['label'], (polys[0][0], polys[0][1]-5), font, font_size, color[label], thickness=1)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
    return img



def mask_image(img, boxes, json_data, widen_range_x, widen_range_y):
    # widen block
    if isinstance(widen_range_x, list) and isinstance(widen_range_y, list):
        boxes = [widen_box(box, np.random.uniform(widen_range_x[0], widen_range_x[1]), np.random.uniform(widen_range_y[0], widen_range_y[1])) for box in boxes]
    else:
        boxes = [widen_box(box, widen_range_x, widen_range_y) for box in boxes]
    
    ls_polys2keep = []
    ls_area2keep = []
    iou_threshold = 0.
    for box_idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        box_pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        p_box = Polygon(box_pts)
        for shape_idx, shape in enumerate(json_data['shapes']):
            if shape_idx in ls_polys2keep:
                continue
            pts = shape['points']
            p_shape = Polygon(pts)
            intersect_area = p_box.intersection(p_shape).area
            if intersect_area / p_shape.area > iou_threshold:
                ls_polys2keep.append(shape_idx)
                pts = [coord for pt in pts for coord in pt]
                poly_xmin = min(pts[::2])
                poly_ymin = min(pts[1::2])
                poly_xmax = max(pts[::2])
                poly_ymax = max(pts[1::2])
                ls_area2keep.append((poly_xmin, poly_ymin, poly_xmax, poly_ymax))

    # mask white all area of image that is not in block
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img.shape[1], xmax)
        ymax = min(img.shape[0], ymax)
        mask[ymin:ymax, xmin:xmax] = 255

    for area2keep in ls_area2keep:
        xmin, ymin, xmax, ymax = area2keep
        xmin = int(max(0, xmin))
        ymin = int(max(0, ymin))
        xmax = int(min(img.shape[1], xmax))
        ymax = int(min(img.shape[0], ymax))
        mask[ymin:ymax, xmin:xmax] = 255

    # mask white
    img[mask == 0] = 255

    # delete all poly that is not in block
    ls_idx2del = [idx for idx, shape in enumerate(json_data['shapes']) if idx not in ls_polys2keep]
    for idx in sorted(ls_idx2del, reverse=True):
        del json_data['shapes'][idx]

    return img, json_data
        

def get_random_area_not_in_block(img_w, img_h, block_boxes):
    ls_block_w = [box[2]-box[0] for box in block_boxes]
    ls_block_h = [box[3]-box[1] for box in block_boxes]
    min_w, max_w = min(ls_block_w), max(ls_block_w)
    min_h, max_h = min(ls_block_h), max(ls_block_h)
    w = np.random.randint(min_w, max_w)
    h = np.random.randint(min_h, max_h)
    
    mask = np.zeros((img_h, img_w))
    for xmin, ymin, xmax, ymax in block_boxes:
        mask[ymin:ymax, xmin:xmax] = 1
    for _ in range(10):
        xmin = np.random.randint(0, img_w-w)
        ymin = np.random.randint(0, img_h-h)
        if np.any(mask[ymin:ymin+h, xmin:xmin+w]==1):
            continue
        else:
            return (xmin, ymin, xmin+w, ymin+h)
    
    return None


def mask_image_poly(img: Image, poly):
    img_w, img_h = img.size
    pts = [coord for pt in poly for coord in pt]
    xmin = np.clip(int(min(pts[::2])), 0, img_w)
    ymin = np.clip(int(min(pts[1::2])), 0, img_h)
    xmax = np.clip(int(max(pts[::2])), 0, img_w)
    ymax = np.clip(int(max(pts[1::2])), 0, img_h)

    img = np.array(img)
    img[ymin:ymax, xmin:xmax] = np.random.randint(240, 255)

    return Image.fromarray(img)



def drop_box_augment(img, json_data, drop_box_percent):
    n_shapes = len(json_data['shapes'])
    idx2drop = np.random.choice(list(range(n_shapes)), size=int(drop_box_percent*n_shapes))

    # drop on image
    for idx in idx2drop:
        shape = json_data['shapes'][idx]
        poly = shape['points']
        img = mask_image_poly(img, poly)

    # drop on json_data
    json_data['shapes'] = [shape for i, shape in enumerate(json_data['shapes']) if i not in idx2drop]

    return img, json_data


def geometric_augment(img, json_data, geometric_transforms):
    ls_orig_poly = [shape['points'] for shape in json_data['shapes']]
    kps = np.array([pt for poly in ls_orig_poly for pt in poly])
    transformed = geometric_transforms(image=np.array(img), keypoints=kps)
    transformed_img = transformed['image']
    transformed_kps = transformed['keypoints']
    transformed_kps = [(int(pt[0]), int(pt[1])) for pt in transformed_kps]
    transformed_polys = [transformed_kps[i:i+4] for i in range(0, len(transformed_kps), 4)]

    # del poly outside of image
    img_w, img_h = transformed_img.shape[1], transformed_img.shape[0]
    ls_idx2del = []
    for i, poly in enumerate(transformed_polys):
        pts = [coord for pt in poly for coord in pt]
        xmin = int(min(pts[::2]))
        ymin = int(min(pts[1::2]))
        xmax = int(max(pts[::2]))
        ymax = int(max(pts[1::2]))
        if xmin < 0 or xmax > img_w or ymin < 0 or ymax > img_h:  # neu box bi loi ra
            xmin = np.clip(xmin, 0, img_w)
            xmax = np.clip(xmax, 0, img_w)
            ymin = np.clip(ymin, 0, img_h)
            ymax = np.clip(ymax, 0, img_h)
            transformed_img[ymin:ymax, xmin:xmax] = np.random.randint(240, 255)  # mask white img
            ls_idx2del.append(i)  # del this box

    # del on json
    new_shapes = []
    for i, shape in enumerate(json_data['shapes']):
        if i not in ls_idx2del:
            shape['points'] = transformed_polys[i]
            new_shapes.append(shape)

    json_data['shapes'] = new_shapes
    img = Image.fromarray(transformed_img)
    return img, json_data



def gen_annotation_for_img(
    mode,
    img_fp, 
    json_fp, 
    ls_disable_label=[], 
    remove_accent=True, 
    augment=False,
    normal_transforms=None,
    geometric_transforms=None,
    gray_prob=0,
    augment_prob=0.3,
    drop_box_prob=0.5,
    drop_box_percent=0.08,
    img_aug_prob=0.4,
    geometric_aug_prob=0.4,
):
    
    img = Image.open(img_fp).convert("RGB")
    json_data = json.load(open(json_fp))

    if mode == 'train' and np.random.rand() < gray_prob:
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.dstack([img, img, img])
        img = Image.fromarray(img)
        # print('grayed')
        
    if augment and np.random.rand() < augment_prob:
        if np.random.rand() < drop_box_prob:   # random drop some boxes
            img, json_data = drop_box_augment(img, json_data, drop_box_percent)

        if np.random.rand() < img_aug_prob:  # noise contrast brightness augment
            img = np.array(img)
            img = normal_transforms(image=img)['image']
            img = Image.fromarray(img)

        if np.random.rand() < geometric_aug_prob:   # shift scale rotate augment
            img, json_data = geometric_augment(img, json_data, geometric_transforms)
    

    words, orig_polys, boxes, labels = [], [], [], []
    img_w, img_h = img.size
    for i, shape in enumerate(json_data['shapes']):
        label = shape['label'] if shape['label'] not in ls_disable_label else 'text'
        words.append(unidecode.unidecode(shape['text'].lower())) if remove_accent else words.append(shape['text'].lower())
            
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
    # check_json_health('/data/tungtx2/huggingface/latest_data_245_final')
    dir = 'VAT_ie_data_new/BW/train'
    cnter = dict(count_all_labels(dir))
    print(json.dumps(cnter, indent=4))
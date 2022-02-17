'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import json
import argparse
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import os
import random

class custom_dataset(Dataset):
    def __init__(self, ann_file):
        self.ann = []
        self.image_paths = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        for ann in self.ann:
            self.image_paths.append(ann['image'])

        self.image_paths = list(set(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        return image_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pseudo_labels_dir', default='pseudo_label_output/', type=str, help='where you saved the pseudo labels')
    parser.add_argument('--bbox_proposal_addr', default='examples/proposals/', type=str)
    parser.add_argument('--output_path', type=str, default='examples/pseudo_labels_coco_style.json')
    parser.add_argument('--image_caption_file', default='examples/image_caption_final.json', type=str)
    parser.add_argument('--root_directory', default='datasets/')

    args = parser.parse_args()
    json_files = [
                args.image_caption_file
               ]
    print(json_files)
    dataset = custom_dataset(json_files)
    coco_anno_all = {}
    coco_anno_all['categories'] = []
    coco_anno_all['images'] = []
    coco_anno_all['annotations'] = []

    categories_to_id = {}
    category_curr_id = 0
    images_to_id = {}
    image_curr_id = 0
    anno_curr_id = 0
    for im_path in tqdm(dataset.image_paths):
        filename = im_path.replace(args.root_directory, args.pseudo_labels_dir)
        filename = os.path.splitext(filename)[0] + '_pseudo_label.pkl'
        image_info_name = im_path.replace(args.root_directory, args.bbox_proposal_addr)
        image_info_name = os.path.splitext(image_info_name)[0] + '_info.pkl'
        with open(filename, 'rb') as f:
            obj_p_labels = pickle.load(f)

        image_item = {}

        with open(image_info_name, 'rb') as f:
            image_info = pickle.load(f)

        image_item['height'] = image_info['ori_shape'][0]
        image_item['width'] = image_info['ori_shape'][1]
        image_item['file_name'] = im_path.replace(args.root_directory, '')
        if not image_item['file_name'] in images_to_id:
            images_to_id[image_item['file_name']] = image_curr_id
            image_curr_id += 1
        image_item['id'] = images_to_id[image_item['file_name']]
        coco_anno_all['images'].append(image_item)

        obj_info = {}
        for obj in obj_p_labels:
            txt, bbox, score = obj
            txt = txt.lower()
            if txt not in obj_info:
                obj_info[txt] = []
            obj_info[txt].append((bbox,score))

        cates = list(obj_info.keys())
        cates = [_.lower() for _ in cates]
        for cls in cates:
            anno_item = {}
            if not cls in categories_to_id:
                categories_to_id[cls] = category_curr_id
                category_curr_id += 1
            box, score = random.choice(obj_info[cls])
            assert box[0] <= image_item['width']+1 and box[2] <= image_item['width']+1
            assert box[1] <= image_item['height']+1 and box[3] <= image_item['height']+1

            anno_item['bbox'] = [int(box[0]), int(box[1]), int(box[2])-int(box[0]), int(box[3])-int(box[1])]
            anno_item['pseudo_score'] = score
            anno_item['area'] = anno_item['bbox'][-1]*anno_item['bbox'][-2]
            anno_item['iscrowd'] = 0
            anno_item["image_id"] = image_item['id']
            anno_item['category_id'] = categories_to_id[cls]
            anno_item['id'] = anno_curr_id
            anno_curr_id += 1
            coco_anno_all['annotations'].append(anno_item)

    for cls in categories_to_id:
        cate_item = {}
        cate_item["supercategory"] = cls
        cate_item["name"] = cls
        cate_item["id"] = categories_to_id[cls]
        coco_anno_all['categories'].append(cate_item)

    with open(args.output_path, 'w') as f:
        json.dump(coco_anno_all, f)





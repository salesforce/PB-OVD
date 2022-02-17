'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import json
from tqdm import tqdm
import argparse
import clip
import torch

def filter_annotation(anno_dict, class_name_to_emb, class_id_to_name, emb_key='ClipEmb'):
    filtered_categories = []
    for item in tqdm(anno_dict['categories']):
        if item['name'] in class_name_to_emb:
            item['embedding'] = {}
            item['embedding'][emb_key] = class_name_to_emb[item['name']]
            filtered_categories.append(item)
    anno_dict['categories'] = filtered_categories

    filtered_images = []
    filtered_annotations = []
    useful_image_ids = set()
    for item in tqdm(anno_dict['annotations']):
        if class_id_to_name[item['category_id']] in class_name_to_emb:
            if not "iscrowd" in item:
                item["iscrowd"] = 0
            filtered_annotations.append(item)
            useful_image_ids.add(item['image_id'])

    for item in tqdm(anno_dict['images']):
        if item['id'] in useful_image_ids:
            if 'file_name' not in item:
                raise Exception("file name should be in anno")
            filtered_images.append(item)
    anno_dict['annotations'] = filtered_annotations
    anno_dict['images'] = filtered_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file_path', type=str, default='examples/pseudo_labels_clipEmb_coco_style.json')
    parser.add_argument('--coco_anno_path', type=str, default='examples/pseudo_labels_coco_style.json')
    args = parser.parse_args()

    prompt_templates =[
        "There is a {category} in the scene.",
        "There is the {category} in the scene.",
        "a photo of a {category} in the scene.",
        "a photo of the {category} in the scene.",
        "a photo of one {category} in the scene.",
        "itap of a {category}.",
        "itap of my {category}.",
        "itap of the {category}.",
        "a photo of a {category}.",
        "a photo of my {category}.",
        "a photo of the {category}.",
        "a photo of one {category}.",
        "a photo of many {category}.",
        "a good photo of a {category}.",
        "a good photo of the {category}.",
        "a bad photo of a {category}.",
        "a bad photo of the {category}.",
        "a photo of a nice {category}.",
        "a photo of the nice {category}.",
        "a photo of a cool {category}.",
        "a photo of the cool {category}.",
        "a photo of a weird {category}.",
        "a photo of the weird {category}.",
        "a photo of a small {category}.",
        "a photo of the small {category}.",
        "a photo of a large {category}.",
        "a photo of the large {category}.",
        "a photo of a clean {category}.",
        "a photo of the clean {category}.",
        "a photo of a dirty {category}.",
        "a photo of the dirty {category}.",
        "a bright photo of a {category}.",
        "a bright photo of the {category}.",
        "a dark photo of a {category}.",
        "a dark photo of the {category}.",
        "a photo of a hard to see {category}.",
        "a photo of the hard to see {category}.",
        "a low resolution photo of a {category}.",
        "a low resolution photo of the {category}.",
        "a cropped photo of a {category}.",
        "a cropped photo of the {category}.",
        "a close-up photo of a {category}.",
        "a close-up photo of the {category}.",
        "a jpeg corrupted photo of a {category}.",
        "a jpeg corrupted photo of the {category}.",
        "a blurry photo of a {category}.",
        "a blurry photo of the {category}.",
        "a pixelated photo of a {category}.",
        "a pixelated photo of the {category}.",
        "a black and white photo of the {category}.",
        "a black and white photo of a {category}",
        "a plastic {category}.",
        "the plastic {category}.",
        "a toy {category}.",
        "the toy {category}.",
        "a plushie {category}.",
        "the plushie {category}.",
        "a cartoon {category}.",
        "the cartoon {category}.",
        "an embroidered {category}.",
        "the embroidered {category}.",
        "a painting of the {category}.",
        "a painting of a {category}.",
    ]

    model, _ = clip.load("ViT-B/32", device='cuda')

    with open(args.coco_anno_path, 'r') as fin:
        coco_anno_all = json.load(fin)

    class_names = {}
    class_id_to_name = {}
    for item in coco_anno_all['categories']:
        name_text_used_for_matching = ' '.join(item['name'].split('_'))
        class_id_to_name[item['id']] = item['name']
        class_names[item['name']] = name_text_used_for_matching

    class_name_list = [n for n in class_names]

    class_lists = [[class_names[n].lower()] for n in class_name_list]

    embeddings = []
    for cls_syns in tqdm(class_lists):
        embedding_cls = []
        for cls_ in cls_syns:
            cls_templates = [template.replace('{category}', cls_) for template in prompt_templates]
            text = clip.tokenize(cls_templates).to('cuda')
            with torch.no_grad():
                embeddings_templates = model.encode_text(text)
                avg_embeddings_templates = torch.mean(embeddings_templates, dim=0)
            embedding_cls.append(avg_embeddings_templates)
        embedding_cls = torch.stack(embedding_cls)
        embedding_cls = torch.mean(embedding_cls, dim=0)
        embeddings.append(embedding_cls)
    embeddings = torch.stack(embeddings)
    embeddings = embeddings.cpu().numpy()


    class_name_to_clipemb = {}

    for c, emb in zip(class_name_list, embeddings.tolist()):
        class_name_to_clipemb[c] = emb

    print(len(class_name_to_clipemb), len(class_names))

    filter_annotation(coco_anno_all, class_name_to_clipemb, class_id_to_name)

    with open(args.output_file_path, 'w') as fout:
        json.dump(coco_anno_all, fout)

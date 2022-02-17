'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_anno_path', type=str, default='examples/pseudo_labels_clipEmb_coco_style.json')
    parser.add_argument('--coco_root', type=str, default="datasets/")
    parser.add_argument('--output_dir', type=str, default="pseudo_label_output/vis")
    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    dataset = CocoDetection(root=args.coco_root, annFile=args.coco_anno_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (images, anns) in enumerate(tqdm(dataloader)):
        image = images[0]
        fig, ax = plt.subplots()
        ax.imshow(image.permute(1, 2, 0))
        image_id = None
        for ann in anns:
            if image_id is None:
                image_id = ann['image_id'].item()
            else:
                assert image_id == ann['image_id'].item()
            cate_name = dataset.coco.cats[ann['category_id'].item()]['name']
            bbox = ann['bbox']
            bbox = [_.item() for _ in bbox]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1], cate_name, style='italic',color='b')
        file_name = dataset.coco.imgs[image_id]['file_name']
        file_name = os.path.basename(file_name)
        plt.axis('off')
        plt.savefig(os.path.join(args.output_dir, file_name))
        plt.clf()

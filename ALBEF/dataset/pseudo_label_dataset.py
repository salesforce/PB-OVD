import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class pseudo_label_dataset(Dataset):
    def __init__(self, ann_file, transform, root_directory, bbox_proposal_addr, max_words=30):        
        self.ann = []
        print(ann_file)
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.pseudo_label_paths = []   

        for ann in self.ann:
            pseudo_label_path = ann['image'].replace(root_directory, bbox_proposal_addr)
            self.pseudo_label_paths.append(pseudo_label_path)
        
        
        #self.image_paths = list(set(self.image_paths))
          
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)   
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
        #pseudo_label_path = ann['proposal_path']
        pseudo_label_path = self.pseudo_label_paths[index]

        return image, caption, pseudo_label_path
    
    

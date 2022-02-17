'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import argparse
import os
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
import numpy as np
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("ALBEF/")
import torch.backends.cudnn as cudnn

from functools import partial
from ALBEF.models.vit import VisionTransformer
from ALBEF.models.xbert import BertConfig, BertModel
from ALBEF.models.tokenization_bert import BertTokenizer

from ALBEF import utils
from ALBEF.dataset import create_dataset, create_sampler, create_loader
import pickle
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches

class VL_Transformer_ITM(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config_bert='',
                 img_size=384
                 ):
        super().__init__()

        bert_config = BertConfig.from_json_file(config_bert)
        self.visual_encoder = VisionTransformer(
            img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        self.itm_head = nn.Linear(768, 2)

    def forward(self, image, text):
        image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        output = self.text_encoder(text.input_ids,
                                   attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   )

        vl_embeddings = output.last_hidden_state[:, 0, :]
        vl_output = self.itm_head(vl_embeddings)
        return vl_output

def vis_det_act(image_, image_relevance, bbox, text, filename, output_dir, bbox_prop = None):
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam
    image_ = image_.unsqueeze(0)
    image = F.interpolate(image_, size=(image_relevance.shape[-2],image_relevance.shape[-1]))
    image = image.squeeze(0)
    image = image.permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    fig, ax = plt.subplots()
    ax.imshow(vis)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    if bbox_prop is not None:
        for bbp in bbox_prop:
            rect = patches.Rectangle((bbp[0], bbp[1]), bbp[2]-bbp[0], bbp[3]-bbp[1], linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
    plt.text(bbox[0]-5, bbox[1]-5, text, color='white', fontsize=15)
    plt.axis('off')
    if not os.path.isdir(os.path.join(output_dir+'/vis')):
        os.makedirs(os.path.join(output_dir+'/vis'))
    plt.savefig(os.path.join(output_dir+'/vis', filename.split('.')[0]+'_{}.png'.format(text.replace('/', '_'))))

def get_activation_map(output, model, image, text_input_mask, block_num, map_size, batch_index):
    loss = output[1].sum()
    image = image.unsqueeze(0)
    text_input_mask = text_input_mask.unsqueeze(0)

    model.zero_grad()
    loss.backward(retain_graph=True)

    with torch.no_grad():
        mask = text_input_mask.view(text_input_mask.size(0),1,-1,1,1)

        grads=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients()
        cams=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map()
        cams = cams[batch_index, :, :, 1:].reshape(image.size(0), 12, -1, map_size, map_size)
        cams = cams * mask
        grads = grads[batch_index, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, map_size, map_size) * mask

        gradcam = cams * grads
        gradcam = gradcam.mean(1)
    return gradcam[0, :, :, :].cpu().detach()

def generate_pseudo_bbox(model, tokenizer, data_loader, object_name_dict, args, block_num, map_size, device):
    num_image_without_proposals = 0
    num_image = 0
 
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    print_freq = 50

    tokenized_dict = {}
    for (k,v_list) in object_name_dict.items():
        tokenized_v_list = []
        for v in v_list:
            value_tmp = tokenizer._tokenize(v)
            value = ' '.join(value_tmp)
            tokenized_v_list.append(value)
        tokenized_dict[k] = tokenized_v_list


    for batch_i, (image, text, proposal_paths) in enumerate(metric_logger.log_every(data_loader, print_freq, '')):
        image = image.to(device, non_blocking=True) 
        text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)
        output = model(image, text_input)
        objects_dict = {} # key is the proposal_path
        objects = []
        # Get object names in every image-caption pair
        for (i, proposal_path) in enumerate(proposal_paths):
            wl = tokenizer._tokenize(text[i])
            tokenizeded_text = ' '.join(wl)
            tokenizeded_text = ' ' + tokenizeded_text + ' '
            objects_for_one = []
            # for every value token, see if there is an exact match
            for (k, v_list) in tokenized_dict.items():
                for v in v_list:
                    left_index = tokenizeded_text.find(' '+v+' ')
                    if left_index != -1:
                        space_count = tokenizeded_text[:(left_index+1)].count(' ')
                        objects_for_one.append((k,v, space_count, space_count+len(v.strip().split(' '))))
            objects.append(objects_for_one)
        
        # Find the best proposal that matches the activation map the most.
        for i, img in enumerate(image):
            filename = proposal_paths[i].split('/')[-1]
            nearest_folder = proposal_paths[i].split('/')[-2]
            # Reading the proposals here.
            _, file_extension = os.path.splitext(proposal_paths[i])
            if file_extension == '':
                proposal_addr = proposal_paths[i]+'.pkl'
                info_addr = proposal_paths[i]+'_info.pkl'
            else:
                proposal_addr = proposal_paths[i].replace(file_extension,'.pkl')
                info_addr = proposal_paths[i].replace(file_extension,'_info.pkl')
            if not os.path.exists(proposal_addr):
                num_image_without_proposals += 1
                print(proposal_addr, "not found")
                continue
            initial_proposals = pickle.load(open(proposal_addr, 'rb'))
            initial_information = pickle.load(open(info_addr, 'rb'))
            im_h, im_w = initial_information['ori_shape'][:2]
            proposals = []
            for p in initial_proposals:
                if p.size != 0:
                    proposals.extend(p)
            if len(proposals) == 0:
                num_image_without_proposals += 1
                continue
            proposals = np.stack(proposals, axis=0)
            prop_boxes = proposals[:,0:4]

            # Get the attention map from gradcam
            act_map = get_activation_map(output[i], model, img, text_input['attention_mask'][i], block_num, map_size, i)
            # Processing the activation map from 
            num_image += 1
            print("Processed " +str(num_image) + " images")
            object_pseudo_list_per_image = []
            for (original_obj_name, replaced_obj_name, obj_i_left, obj_i_right) in objects[i]:
                score_max = -1
                best_proposal = [0, 0, 0, 0]
                act_map_obj = act_map[obj_i_left]
                if obj_i_right - obj_i_left > 1:
                    for obj_i in range(obj_i_left+1, obj_i_right):
                        act_map_obj += act_map[obj_i]
                act_map_obj = F.interpolate(act_map_obj.unsqueeze(0).unsqueeze(0), size=(im_h, im_w)).cpu().numpy()
                act_map_obj = act_map_obj.squeeze()
                act_map_obj= (act_map_obj - act_map_obj.min()) / (act_map_obj.max() - act_map_obj.min())
                value_max = np.max(act_map_obj[:])
                act_map_obj[act_map_obj < value_max * 0.5] = 0.0  # ignore insignificant responses


                for bi, bb in enumerate(prop_boxes):
                    bb_tmp = np.copy(bb)  # xmin, ymin, xmax, ymax

                    area = float(bb_tmp[2] - bb_tmp[0]) * float(bb_tmp[3] - bb_tmp[1])

                    if bb_tmp[0] < 0 or bb_tmp[1] < 0 or bb_tmp[2] > act_map_obj.shape[1] or bb_tmp[3] > act_map_obj.shape[0]:
                        continue


                    det_score = act_map_obj[int(bb_tmp[1]):int(bb_tmp[3]), int(bb_tmp[0]):int(bb_tmp[2])]
                    if len(det_score) == 0 or area == 0:
                        continue
                    det_score = det_score.sum() / area ** 0.5
                    if det_score > score_max:
                        score_max = det_score
                        best_proposal = [bb[0], bb[1], bb[2], bb[3]]

                object_pseudo_list_per_image.append((original_obj_name, best_proposal, score_max)) # image_addr_pseudo_label.pkl: [("person", [xmin,..,], det_score),("person",),()]
                vis_det_act(img, act_map_obj, best_proposal, original_obj_name, nearest_folder+'_'+filename,
                           args.output_dir, prop_boxes)
            if proposal_paths[i] not in objects_dict.keys():
                objects_dict[proposal_paths[i]]= object_pseudo_list_per_image
            else:
                objects_dict[proposal_paths[i]].extend(object_pseudo_list_per_image)

        for (k, v) in objects_dict.items():
            output_addr = k.replace(args.bbox_proposal_addr,args.output_dir)
            _, file_extension = os.path.splitext(output_addr)
            if file_extension == '':
                output_addr = output_addr+'_pseudo_label.pkl'
            else:
                output_addr = output_addr.replace(file_extension,'_pseudo_label.pkl')

            if not os.path.isdir(os.path.dirname(output_addr)):
                os.makedirs(os.path.dirname(output_addr))

            with open(output_addr, 'wb') as fp:
                pickle.dump(v, fp)

def main(args, config):   
    
    device = torch.device(args.device)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = [create_dataset('pseudolabel', config, args.root_directory, args.bbox_proposal_addr)]

    data_loader = create_loader(datasets, [None],batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model Initialization#### 
    print("Creating model......")
    bert_config_path = 'ALBEF/configs/config_bert.json'
    model_path = args.model_path
    img_size = 256
    map_size = 16
    model = VL_Transformer_ITM(text_encoder='bert-base-uncased', config_bert=bert_config_path, img_size=img_size)
    model = model.to(device)

    #### Load the Model####
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    for key in list(state_dict.keys()): # adjust different names in pretrained checkpoint
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]

    print("Start loading form the checkpoint......")
    msg = model.load_state_dict(state_dict,strict=False)
    assert len(msg.missing_keys) == 0

    model.eval()
    block_num = 8

    model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True 
    print("Loading object name dictionary....")
    with open(args.object_dict, 'r') as fp:
        object_name_dict = json.load(fp)
    print("Start generating pseudo bbox...")
    start_time = time.time()
    generate_pseudo_bbox(model, tokenizer, data_loader, object_name_dict, args, block_num, map_size, device)
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='ALBEF/configs/Pretrain.yaml')
    parser.add_argument('--model_path', default='examples/ALBEF.pth')
    parser.add_argument('--root_directory', default='datasets/')
    parser.add_argument('--output_dir', default='pseudo_label_output/')
    parser.add_argument('--object_dict', default='examples/object_vocab.json')
    parser.add_argument('--bbox_proposal_addr', default='examples/proposals/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)

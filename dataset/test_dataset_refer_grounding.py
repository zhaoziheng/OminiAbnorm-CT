import json
import os
from PIL import Image
import random
import numpy as np
import cv2
from einops import repeat, rearrange

import torch
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

# Add parent directory to path to find the utils module (when debugging in this script)
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.distribute import main_print

def format_data(image, instruction):
    system_message = "You are a helpful medical assistant."
    
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": 262144,
                    "max_pixels": 262144,
                },
                {
                    "type": "text",
                    "text": instruction,
                }
            ],
        }
    ]

class Refer_Grounding_Dataset(Dataset):
    
    def __init__(self, 
                 # Seg Data
                 deeplesion_json=None,
                 radiopedia_json=None,
                 other_data_json=None,
                 deeplesion_weight=0,
                 radiopedia_weight=0,
                 other_data_weight=0,
                 dataset_config_dict='./dataset/config/other_datasets.json',
                 ):
        
        self.prepare_refer_seg_data(
            deeplesion_json, 
            radiopedia_json, 
            other_data_json, 
            deeplesion_weight, 
            radiopedia_weight, 
            other_data_weight, 
            dataset_config_dict
        )
        
        self.context_range = 4
        
        main_print(f"** DATA ** Refer Grounding Samples: {len(self.other_data_idx_ls)+len(self.radiopedia_idx_ls)+len(self.deeplesion_idx_ls)}")
        
    def prepare_refer_seg_data(
        self, 
        deeplesion_json, 
        radiopedia_json, 
        other_data_json,
        deeplesion_weight,
        radiopedia_weight,
        other_data_weight,
        dataset_config_dict
        ):
        
        # Data and Sampling Weight
        
        assert (deeplesion_json is None) or deeplesion_weight>=0 
        assert (radiopedia_json is None) or radiopedia_weight>=0
        assert (other_data_json is None) or other_data_weight>=0
        
        if deeplesion_json is not None:
            with open(deeplesion_json, 'r') as f:
                self.deeplesion_data_dict = json.load(f)
            self.deeplesion_idx_ls = list(self.deeplesion_data_dict.keys())
        else:
            self.deeplesion_idx_ls = []
            deeplesion_weight = 0
                
        if radiopedia_json is not None:
            with open(radiopedia_json, 'r') as f:
                self.radiopedia_data_dict = json.load(f)
            # 每个group取一个pos
            # (前33%的sample)整个image取一个neg    
            self.radiopedia_idx_ls = []
            for idx, (datum_idx, datum) in enumerate(self.radiopedia_data_dict.items()):
                if 'abnormality_group' in datum:
                    for group_id, abnormality_group in datum['abnormality_group'].items():
                        if len(abnormality_group[0]['referring_prompt']) > 0:
                            self.radiopedia_idx_ls.append(f'{datum_idx}/{group_id}/pos')
                if idx <= 0.33 * len(self.radiopedia_data_dict):
                    self.radiopedia_idx_ls.append(f'{datum_idx}/0/neg')
        else:
            self.radiopedia_idx_ls = []
            radiopedia_weight=0
        
        if other_data_json is not None:
            with open(other_data_json, 'r', encoding='utf-8') as f:
                self.other_data_dict = json.load(f)
            self.other_data_idx_ls = list(self.other_data_dict.keys())
        else:
            self.other_data_idx_ls = []
            other_data_weight=0
            
        if deeplesion_weight + radiopedia_weight + other_data_weight == 0:
            deeplesion_weight = len(self.deeplesion_idx_ls)
            radiopedia_weight = len(self.radiopedia_idx_ls)
            other_data_weight = len(self.other_data_idx_ls)
        self.refer_seg_data_source_weight = [deeplesion_weight, radiopedia_weight, other_data_weight]
            
        with open(dataset_config_dict, 'r') as f:
            dataset_config_dict = json.load(f)
        self.other_data_windows = dataset_config_dict['other_data_windows']
    
    def _pad_truncate_and_resize(self, stack_img, mask, target_h, target_w, pre_padding_len, post_padding_len):
        """
        pad image to H W D
        pad mask to H W
        """
        h, w, d = stack_img.shape
        
        padded_img = []
        
        # depth padding or truncate
        if pre_padding_len > 0:
            padded_img.append(np.zeros(shape=(h, w, pre_padding_len)))   # e.g. pre_padding_len = 2, then pad 2
        if pre_padding_len < 0:
            stack_img = stack_img[:, :, (-1)*pre_padding_len:]  # e.g. pre_padding_len = -2, then [2:]
            
        padded_img.append(stack_img)
        
        if post_padding_len > 0:
            padded_img.append(np.zeros(shape=(h, w, post_padding_len)))
            
        padded_img = np.concatenate(padded_img, axis=-1)
        
        if post_padding_len < 0:
            padded_img = padded_img[:, :, :post_padding_len]   # e.g. post_padding_len = -2, then [:-2]
        
        # h w padding
        
        if h > w:
            left_padding_len = (h - w) // 2
            right_padding_len = h - w - left_padding_len
            padded_img = np.pad(padded_img, ((0, 0), (left_padding_len, right_padding_len), (0, 0)), 'constant')
            padded_msk = np.pad(mask, ((0, 0), (left_padding_len, right_padding_len)), 'constant')
        elif w > h:
            up_padding_len = (w - h) // 2
            down_padding_len = w - h - up_padding_len
            padded_img = np.pad(padded_img, ((up_padding_len, down_padding_len), (0, 0), (0, 0)), 'constant')
            padded_msk = np.pad(mask, ((up_padding_len, down_padding_len), (0, 0)), 'constant')
        else:
            padded_msk = mask
        
        # Resize the mask (cv2.resize takes (width, height) not (height, width))
        resized_msk = cv2.resize(padded_msk, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # Resize each slice of the 3D image stack
        resized_slices = []
        for d_idx in range(padded_img.shape[2]):
            slice_2d = padded_img[:, :, d_idx]
            resized_slice = cv2.resize(slice_2d, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized_slices.append(resized_slice)
        # Stack the resized slices back together
        resized_img = np.stack(resized_slices, axis=2)  # stack along the depth dimension
        
        return resized_img, resized_msk
    
    def _get_pil_image(self, img):
        # convert h*w tensor to pil_image
                
        img_min, img_max = img.min(), img.max()
        img_scaled = (img - img_min) / (img_max - img_min + 1e-10)
    
        # Convert to uint8 (0-255)
        img_uint8 = (img_scaled * 255).astype(np.uint8)
        
        # Create PIL image
        if img.ndim == 3 and img.shape[2] == 3:
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)  # Add this line
            pil_img = Image.fromarray(img_uint8, mode='RGB')  # 'RGB' for color images
        else:
            pil_img = Image.fromarray(img_uint8, mode='L')  # 'L' for grayscale
        
        return pil_img
    
    def _get_deeplesion_item(self, idx):
        raise NotImplementedError("TODO DeepLesion data is not available now.")
        
    def _get_radiopedia_item(self, idx):
        datum_idx, group_id, pos_or_neg = idx.split('/')
        is_pos = (pos_or_neg == 'pos')
        datum = self.radiopedia_data_dict[datum_idx]
        suffix = 'jpeg' if os.path.exists(os.path.join(datum['image'], '0.jpeg')) else 'jpg'
        
        # load image
        img_ls = []
        center_img = cv2.imread(os.path.join(datum['image'], f'0.{suffix}'), cv2.IMREAD_GRAYSCALE)
        if center_img is None:
            raise ValueError(f"Fail to load {datum['image']}/0.{suffix}")
        for file_name in datum['context']:
            if os.path.exists(os.path.join(datum['image'], f'{file_name}.{suffix}')):
                img = cv2.imread(os.path.join(datum['image'], f'{file_name}.{suffix}'), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    img = np.zeros_like(center_img)
                    print(f"Fail to load {datum['image']}/{file_name}.{suffix}")
            else:
                img = np.zeros_like(center_img)
                print(f"Not exist: {datum['image']}/{file_name}.{suffix}")
            img = (img - np.mean(img)) / (np.std(img) + 1e-10)
            # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if img.shape != center_img.shape:
                img = cv2.resize(img, (center_img.shape[1], center_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            img_ls.append(img)
        stack_img = np.stack(img_ls, axis=-1)
        
        # load pil image
        pil_img = self._get_pil_image(center_img)
        
        # load referring and mask
        mask = np.zeros_like(center_img)
        if not is_pos:
            # randomly choose a non-existing referring
            referring = datum['abnormality_group'][group_id][0]['negative_referring_prompt'][0] # NOTE take the 1st negative refer of the 1st abnormality in the 1st group
            # pil_img_w_prmpt = pil_img
            answer = "I don't see any relevant abnormalities on the image."
            label_ls = description_ls = ['negative']
            type_ls_ls = [['negative']]
        else:
            # Get all abnormalities in this group
            abnormality_in_this_group = datum['abnormality_group'][group_id]

            referring = datum['abnormality_group'][group_id][0]['referring_prompt'][0]  # NOTE take the 1st refer of the 1st abnormality in the group
            
            # Create mask for all abnormalities in the group
            for abnormality in abnormality_in_this_group:
                mask_path = os.path.join(datum['mask'], f'{abnormality["id"]}.jpeg')
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(datum['mask'], f'{abnormality["id"]}.jpg')
                if not os.path.exists(mask_path):
                    raise ValueError(f"Cant not find {datum['mask']}/{abnormality['id']}")
                abn_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if abn_mask is None:
                    raise ValueError(f"Fail to load {mask_path}")
                mask = np.maximum(mask, abn_mask)
            
            # # load the vis_prmpt_image
            # tmp = '_'.join(str(tmp['id']) for tmp in abnormality_in_this_group)   # 0 1 2 -> 0_1_2
            # img_w_prmpt = cv2.imread(os.path.join(datum['image'], 'visual_instruction', f"bbox_{tmp}.png"), cv2.IMREAD_GRAYSCALE)
            # pil_img_w_prmpt = self._get_pil_image(img_w_prmpt)
            
            # take the description of the 1st abnormality
            answer = abnormality_in_this_group[0]['description']
            
            label_ls = []
            description_ls = []
            type_ls_ls = []
            for abnormality in abnormality_in_this_group:
                if len(abnormality['label']) > 0 and len(label_ls) == 0:
                    label_ls = [abnormality['label']]
                if len(abnormality['description']) > 0 and len(description_ls) == 0:
                    description_ls = [abnormality['description']] 
                if 'category' in abnormality and len(abnormality['category']) > 0 and len(type_ls_ls) == 0:
                    type_ls_ls = [abnormality['category']]
            if len(label_ls) == 0:
                label_ls = ['']
            if len(description_ls) == 0:
                description_ls = ['']
            if len(type_ls_ls) == 0:
                type_ls_ls = [['']]
        
        # Find the negative number in datum['context']
        pre_len = len([num for num in datum['context'] if num < 0])
        post_len = len([num for num in datum['context'] if num > 0])
        pre_padding_len = self.context_range - pre_len
        post_padding_len = self.context_range - post_len
        
        # padding and resize
        img, msk = self._pad_truncate_and_resize(stack_img, mask, 512, 512, pre_padding_len, post_padding_len)  # (h w d), (h w)

        return pil_img, img, msk, referring, answer, label_ls, description_ls, type_ls_ls
    
    def _get_other_data_item(self, idx):
        raise NotImplementedError("TODO Other data is not available now.")
    
    def __len__(self):
        return len(self.deeplesion_idx_ls)+len(self.radiopedia_idx_ls)+len(self.other_data_idx_ls)
    
    def __getitem__(self, idx):

        if idx < len(self.deeplesion_idx_ls):
            sample_idx = self.deeplesion_idx_ls[idx]
            dataset_name = 'deeplesion'
            pil_img, img, msk, referring, llm_output, label_ls, description_ls, type_ls_ls = self._get_deeplesion_item(sample_idx)

        elif idx < len(self.deeplesion_idx_ls) + len(self.radiopedia_idx_ls):
            sample_idx = self.radiopedia_idx_ls[idx-len(self.deeplesion_idx_ls)]
            dataset_name = 'radiopedia'
            pil_img, img, msk, referring, llm_output, label_ls, description_ls, type_ls_ls = self._get_radiopedia_item(sample_idx)
        
        else:
            sample_idx = self.other_data_idx_ls[idx-len(self.deeplesion_idx_ls)-len(self.radiopedia_idx_ls)]
            dataset_name = '_'.join(sample_idx.split('_')[:-2])
            pil_img, img, msk, referring, llm_output, label_ls, description_ls, type_ls_ls = self._get_other_data_item(sample_idx)
        
        # make sure msk is binary after rescaling
        msk = np.where(msk>0.5, 1.0, 0.0)
        
        # Image augmentation
        img = rearrange(img, 'h w d -> d h w')  # add channel to use monai transform
        msk = repeat(msk, 'h w -> c h w', c=1)
            
        return {
            'vllm_input': format_data(pil_img, referring),
            # input to seg model
            'seg_img': img, 
            'seg_gt': msk,
            # draw prompt on this after seg
            # 'pil_img': pil_img,
            'answer_gt': llm_output,
            'dataset_name': dataset_name,
            'sample_id': sample_idx,
            'type_ls_ls': type_ls_ls,
            'label_ls': label_ls,
            'description_ls': description_ls
        }
        
class Refer_Grounding_Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples: list):
        batch_vllm_raw_input = [example['vllm_input'] for example in examples]
        
        # Get the texts and images, similar to training collate_fn
        texts = [self.processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in batch_vllm_raw_input]
        image_inputs = [process_vision_info(example)[0] for example in batch_vllm_raw_input]
        
        # Tokenize the texts and process the images
        batch_vllm_input = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True, truncation=True)
        
        # in and gt for seg task
        bs_seg_img = []
        bs_seg_gt = []
        for example in examples:
            if isinstance(example['seg_img'], torch.Tensor):
                bs_seg_img.append(example['seg_img'])
            else:
                bs_seg_img.append(torch.from_numpy(example['seg_img']))
            if isinstance(example['seg_gt'], torch.Tensor):
                bs_seg_gt.append(example['seg_gt'])
            else:
                bs_seg_gt.append(torch.from_numpy(example['seg_gt']))       
        bs_seg_img = torch.stack(bs_seg_img, dim=0)
        bs_seg_gt = torch.stack(bs_seg_gt, dim=0)
        
        batch = {
            'vllm_input': batch_vllm_input,
            'vllm_raw_input': batch_vllm_raw_input,
            'seg_img': bs_seg_img,
            'seg_gt': bs_seg_gt,
            # 'pil_img': [example['pil_img'] for example in examples],
            'answer_gt': [example['answer_gt'] for example in examples],
            'sample_id': [example['sample_id'] for example in examples],
            'dataset_name': [example['dataset_name'] for example in examples],
            'label_ls': [example['label_ls'] for example in examples],
            'description_ls': [example['description_ls'] for example in examples],
            'type_ls_ls': [example['type_ls_ls'] for example in examples],
        }
        
        return batch

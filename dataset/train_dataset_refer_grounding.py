import json
import os
from PIL import Image
import random
import numpy as np
import cv2
from einops import repeat
import traceback

from torch.utils.data import Dataset

# Add parent directory to path to find the utils module (when debugging in this script)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.distribute import main_print
from utils.online_prmpt_generation import draw_bounding_box_w_random_shift

def format_two_round_data(image, instruction, seg_output, llm_output):
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
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": '<SEG>'
                }
            ],
        },
        {
            "role": "tool",
            "name": "segmentation_tool",
            "content": [
                {
                    "type": "image",
                    "image": seg_output,
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": llm_output,
                }
            ],
        },
    ]

class Refer_Grounding_Two_Round_Dataset(Dataset):
    
    def __init__(self, 
                 # Seg Data
                 deeplesion_json=None,
                 radiopedia_json=None,
                 other_data_json=None,
                 deeplesion_weight=0,
                 radiopedia_weight=0,
                 other_data_weight=0,
                 negative_prompt_ratio=0.3,
                 dataset_config_dict='./dataset/config/other_datasets.json',
                 epoch_quota=100000, # Quota for this dataset in an epoch
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
        
        self.epoch_quota = epoch_quota
        
        self.context_range = 4
        
        self.negative_prompt_ratio = negative_prompt_ratio  # refer to something not on this image
        
        main_print(f"** DATA ** Refer Grounding Samples: {len(self.other_data_idx_ls)+len(self.radiopedia_idx_ls)+len(self.deeplesion_idx_ls)}; Samples in One Epoch: {epoch_quota}")
        
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
        
        from monai.transforms import (
            Compose,
            RandRotated,
            RandShiftIntensityd,
            RandRotate90d,
            RandZoomd,
            RandGaussianSmoothd,
            RandGaussianNoised,
            RandGaussianSharpend,
            RandScaleIntensityd,
            # RandSimulateLowResolutiond,
            RandAdjustContrastd
        )
        
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
                radiopedia_data_dict = json.load(f)
            # All referring prompts in a pool for negative sampling
            self.radiopedia_referring_pool = []
            # Process radiopedia data to organize by group
            self.radiopedia_data_dict = {}
            for idx, datum in radiopedia_data_dict.items():
                if 'abnormality' not in datum:
                    continue
                self.radiopedia_data_dict[idx] = {'image':datum['image'], 'mask':datum['mask'], 'context':datum['context'], 'abnormality_group':{}}
                abnormality_by_id = {abn['id']: abn for abn in datum['abnormality'] if 'id' in abn}
                for group_id, group_info in datum['abnormality_group'].items():
                    self.radiopedia_data_dict[idx]['abnormality_group'][group_id] = [abnormality_by_id[abn_id] for abn_id in group_info['id']]
                for abnormality in datum['abnormality']:
                    for i in range(1, 6):
                        if f'referring_prompt_{i}' in abnormality:
                            self.radiopedia_referring_pool.append(abnormality[f'referring_prompt_{i}'])
            self.radiopedia_idx_ls = list(self.radiopedia_data_dict.keys())
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
            
        # Augmentation
            
        self.augmentator = None
        with open(dataset_config_dict, 'r') as f:
            dataset_config_dict = json.load(f)
        self.other_data_windows = dataset_config_dict['other_data_windows']
        config = dataset_config_dict['augmentation']
        aug_ls = []
        if 'RandRotate90' in config:
            aug_ls.append(RandRotate90d(
                    keys=["image", "label"], 
                    spatial_axes=(1, 2),
                    max_k=config['RandRotate90']['max_k'],
                    prob=config['RandRotate90']['prob'],
                )
            )
        if 'RandRotate' in config:
            aug_ls.append(RandRotated(
                    keys=["image", "label"], 
                    range_x=config['RandRotate']['range'],
                    prob=config['RandRotate']['prob'],
                )
            )
        if 'RandZoom' in config:
            aug_ls.append(RandZoomd(
                    keys=["image", "label"], 
                    mode=['area', 'nearest'],
                    min_zoom=config['RandZoom']['min_zoom'],
                    max_zoom=config['RandZoom']['max_zoom'],
                    prob=config['RandZoom']['prob'],
                )
            )
        if 'RandGaussianNoise' in config:
            aug_ls.append(RandGaussianNoised(
                    keys=["image"],
                    mean=config['RandGaussianNoise']['mean'],
                    std=config['RandGaussianNoise']['std'],
                    prob=config['RandGaussianNoise']['prob'],
                )
            )
        if 'RandGaussianSmooth' in config:
            aug_ls.append(
                RandGaussianSmoothd(
                    keys=['image'],
                    sigma_x=config['RandGaussianSmooth']['sigma'],
                    sigma_y=(0, 0),
                    sigma_z=(0, 0),
                    prob=config['RandGaussianSmooth']['prob'],
                )
            )
        if 'RandScaleIntensity' in config:
            aug_ls.append(
                RandScaleIntensityd(
                    keys=['image'],
                    factors=config['RandScaleIntensity']['factors'],
                    prob=config['RandScaleIntensity']['prob']
                )
            )
        if 'RandAdjustContrast' in config:
            aug_ls.append(
                RandAdjustContrastd(
                    keys=['image'],
                    # retain_stats=config['RandAdjustContrast']['retain_stats'],
                    # invert_image=config['RandAdjustContrast']['invert_image'],
                    gamma=config['RandAdjustContrast']['gamma'],
                    prob=config['RandAdjustContrast']['prob']
                )
            )
        if len(aug_ls) > 0:
            self.augmentator = Compose(aug_ls)
            
        self.random_context_drop = config['RandDropContext']
    
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
        datum = self.radiopedia_data_dict[idx]
        suffix = 'jpeg' if os.path.exists(os.path.join(datum['image'], '0.jpeg')) else 'jpg'
        
        # load image
        center_img = cv2.imread(os.path.join(datum['image'], f'0.{suffix}'), cv2.IMREAD_GRAYSCALE)
        if center_img is None:
            raise ValueError(f"Fail to load {datum['image']}/0.{suffix}")
        
        # load pil image
        pil_img = self._get_pil_image(center_img)
        
        # a negative sample
        if random.random() < self.negative_prompt_ratio:
            # randomly choose a referring from another datum
            referring = random.choice(self.radiopedia_referring_pool)
            pil_img_w_prmpt = pil_img
            answer = "I don't see any relevant abnormalities on the image."
            mask = np.zeros_like(center_img)
        
        # a positive sample    
        else:
            # Choose a group of abnormality
            chosen_group = random.choice(list(datum['abnormality_group'].keys()))
            
            # Get all abnormalities in this group
            abnormality_in_this_group = datum['abnormality_group'][chosen_group]
            
            # 1. Create mask for all abnormalities in the group
            # 2. Find all potential referring prompt
            # 3. Find all potential gt answer / finding
            mask = np.zeros_like(center_img)
            all_referrings = []
            all_gts = []
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
                
                for i in range(1, 6):
                    if f'referring_prompt_{i}' in abnormality:
                        all_referrings.append(abnormality[f'referring_prompt_{i}'])
                
                if 'rewritten_findings' in abnormality:       
                    all_gts += abnormality['rewritten_findings'] 
                all_gts += [abnormality['description']]
                        
            if len(all_referrings) == 0:
                raise ValueError(f"Cannot find any referring in abnormaliy group {chosen_group} in datum {idx}")
            
            if len(all_gts) == 0:
                raise ValueError(f"Cannot find any gt findings in abnormaliy group {chosen_group} in datum {idx}")
            
            if len(all_gts) == 0:
                raise ValueError(f"Cannot find any gt findings in abnormaliy group {chosen_group} in datum {idx}")
                        
            # Check if there's at least one non-empty string in all_gts
            if not any(gt.strip() for gt in all_gts):
                raise ValueError(f"All gt findings in abnormality group {chosen_group} in datum {idx} are empty")
            
            # randomly choose a referring from this group
            referring = random.choice(all_referrings)
            
            # randomly choose a non-empty gt answer from this group
            while len(all_gts) > 0:
                answer = random.choice(all_gts)
                if answer.strip() != "":
                    break
                all_gts.remove(answer)
            
            # NOTE Simulate prompt with some extents of shift to make model more robust
            img_w_prmpt = draw_bounding_box_w_random_shift(center_img, mask, color=(0, 0, 255), thickness=2, shift_range=0.25)
            pil_img_w_prmpt = self._get_pil_image(img_w_prmpt)
            
        return pil_img, None, pil_img_w_prmpt, None, referring, answer, datum['image']
    
    def _get_other_data_item(self, idx):
        raise NotImplementedError("TODO Other data is not available now.")

    def __len__(self):
        return self.epoch_quota
    
    def __getitem__(self, idx):
        
        while True:
            
            try:
                # 随机抽取一个sample
                chosen = random.choices(['deeplesion', 'radiopedia', 'Others'], weights=self.refer_seg_data_source_weight, k=1)[0]
                
                if chosen == 'deeplesion':
                    sample_idx = random.choice(self.deeplesion_idx_ls)
                    pil_img, img, pil_img_w_prmpt, msk, referring, llm_output, img_path = self._get_deeplesion_item(sample_idx)

                elif chosen == 'radiopedia':
                    sample_idx = random.choice(self.radiopedia_idx_ls)
                    pil_img, img, pil_img_w_prmpt, msk, referring, llm_output, img_path = self._get_radiopedia_item(sample_idx)
                    
                else:
                    sample_idx = random.choice(self.other_data_idx_ls)
                    pil_img, img, pil_img_w_prmpt, msk, referring, llm_output, img_path = self._get_other_data_item(sample_idx)
                    
                break

            except SystemExit:
                exit()
            except:
                # record bugs in loading data
                traceback_info = traceback.format_exc()
                print(f'*** {sample_idx} *** {chosen} ***\n')
                print(traceback_info)
            
        return {
            'task_type': 'refer_grounding_two_round',
            'vlm_data': format_two_round_data(pil_img, referring, pil_img_w_prmpt, llm_output),
            # 'seg_img': img,
            # 'seg_gt': msk,
            'img_path': img_path
            }

def format_one_round_data(image, instruction):
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
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": '<SEG>'
                }
            ],
        }
    ]

class Refer_Grounding_One_Round_Dataset(Dataset):
    def __init__(self, 
                 # Seg Data
                 deeplesion_json=None,
                 radiopedia_json=None,
                 other_data_json=None,
                 deeplesion_weight=0,
                 radiopedia_weight=0,
                 other_data_weight=0,
                 negative_prompt_ratio=0.3,
                 dataset_config_dict='./dataset/config/other_datasets.json',
                 epoch_quota=100000, # Quota for this dataset in an epoch
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
        
        self.epoch_quota = epoch_quota
        
        self.context_range = 4
        
        self.negative_prompt_ratio = negative_prompt_ratio  # refer to something not on this image
        
        main_print(f"** DATA ** Refer Grounding Samples: {len(self.other_data_idx_ls)+len(self.radiopedia_idx_ls)+len(self.deeplesion_idx_ls)}; Samples in One Epoch: {epoch_quota}")
        
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
        
        from monai.transforms import (
            Compose,
            RandRotated,
            RandShiftIntensityd,
            RandRotate90d,
            RandZoomd,
            RandGaussianSmoothd,
            RandGaussianNoised,
            RandGaussianSharpend,
            RandScaleIntensityd,
            # RandSimulateLowResolutiond,
            RandAdjustContrastd
        )
        
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
                radiopedia_data_dict = json.load(f)
            # All referring prompts in a pool for negative sampling
            self.radiopedia_referring_pool = []
            # Process radiopedia data to organize by group
            self.radiopedia_data_dict = {}
            for idx, datum in radiopedia_data_dict.items():
                if 'abnormality' not in datum:
                    continue
                self.radiopedia_data_dict[idx] = {'image':datum['image'], 'mask':datum['mask'], 'context':datum['context'], 'abnormality_group':{}}
                abnormality_by_id = {abn['id']: abn for abn in datum['abnormality'] if 'id' in abn}
                for group_id, group_info in datum['abnormality_group'].items():
                    self.radiopedia_data_dict[idx]['abnormality_group'][group_id] = [abnormality_by_id[abn_id] for abn_id in group_info['id']]
                for abnormality in datum['abnormality']:
                    for i in range(1, 6):
                        if f'referring_prompt_{i}' in abnormality:
                            self.radiopedia_referring_pool.append(abnormality[f'referring_prompt_{i}'])
            self.radiopedia_idx_ls = list(self.radiopedia_data_dict.keys())
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
            
        # Augmentation
            
        self.augmentator = None
        with open(dataset_config_dict, 'r') as f:
            dataset_config_dict = json.load(f)
        self.other_data_windows = dataset_config_dict['other_data_windows']
        config = dataset_config_dict['augmentation']
        aug_ls = []
        if 'RandRotate90' in config:
            aug_ls.append(RandRotate90d(
                    keys=["image", "label"], 
                    spatial_axes=(1, 2),
                    max_k=config['RandRotate90']['max_k'],
                    prob=config['RandRotate90']['prob'],
                )
            )
        if 'RandRotate' in config:
            aug_ls.append(RandRotated(
                    keys=["image", "label"], 
                    range_x=config['RandRotate']['range'],
                    prob=config['RandRotate']['prob'],
                )
            )
        if 'RandZoom' in config:
            aug_ls.append(RandZoomd(
                    keys=["image", "label"], 
                    mode=['area', 'nearest'],
                    min_zoom=config['RandZoom']['min_zoom'],
                    max_zoom=config['RandZoom']['max_zoom'],
                    prob=config['RandZoom']['prob'],
                )
            )
        if 'RandGaussianNoise' in config:
            aug_ls.append(RandGaussianNoised(
                    keys=["image"],
                    mean=config['RandGaussianNoise']['mean'],
                    std=config['RandGaussianNoise']['std'],
                    prob=config['RandGaussianNoise']['prob'],
                )
            )
        if 'RandGaussianSmooth' in config:
            aug_ls.append(
                RandGaussianSmoothd(
                    keys=['image'],
                    sigma_x=config['RandGaussianSmooth']['sigma'],
                    sigma_y=(0, 0),
                    sigma_z=(0, 0),
                    prob=config['RandGaussianSmooth']['prob'],
                )
            )
        if 'RandScaleIntensity' in config:
            aug_ls.append(
                RandScaleIntensityd(
                    keys=['image'],
                    factors=config['RandScaleIntensity']['factors'],
                    prob=config['RandScaleIntensity']['prob']
                )
            )
        if 'RandAdjustContrast' in config:
            aug_ls.append(
                RandAdjustContrastd(
                    keys=['image'],
                    # retain_stats=config['RandAdjustContrast']['retain_stats'],
                    # invert_image=config['RandAdjustContrast']['invert_image'],
                    gamma=config['RandAdjustContrast']['gamma'],
                    prob=config['RandAdjustContrast']['prob']
                )
            )
        if len(aug_ls) > 0:
            self.augmentator = Compose(aug_ls)
            
        self.random_context_drop = config['RandDropContext']
    
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
        datum = self.radiopedia_data_dict[idx]
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
        img = cv2.imread(os.path.join(datum['image'], f'0.{suffix}'), cv2.IMREAD_GRAYSCALE)
        pil_img = self._get_pil_image(img)
        
        # a negative sample
        if random.random() < self.negative_prompt_ratio:
            # randomly choose a referring from another datum
            referring = random.choice(self.radiopedia_referring_pool)
            # pil_img_w_prmpt = pil_img
            # answer = "I don't see any relevant abnormalities on the image."
            mask = np.zeros_like(img)
        
        # a positive sample    
        else:
            # Choose a group of abnormality
            chosen_group = random.choice(list(datum['abnormality_group'].keys()))
            
            # Get all abnormalities in this group
            abnormality_in_this_group = datum['abnormality_group'][chosen_group]
            
            # 1. Create mask for all abnormalities in the group
            # 2. Find all potential referring prompt
            # 3. Find all potential gt answer / finding
            mask = np.zeros_like(img)
            all_referrings = []
            # all_gts = []
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
                
                for i in range(1, 6):
                    if f'referring_prompt_{i}' in abnormality:
                        all_referrings.append(abnormality[f'referring_prompt_{i}'])
                        
                # all_gts += abnormality['rewritten_findings'] 
                # all_gts += [abnormality['description']]
                        
            if len(all_referrings) == 0:
                raise ValueError(f"Cannot find any referring in abnormaliy group {chosen_group} in datum {idx}")
            
            # randomly choose a referring from this group
            referring = random.choice(all_referrings)

        # Find the negative number in datum['context']
        pre_len = len([num for num in datum['context'] if num < 0])
        post_len = len([num for num in datum['context'] if num > 0])
        pre_padding_len = self.context_range - pre_len
        post_padding_len = self.context_range - post_len
        
        # padding and resize
        img, msk = self._pad_truncate_and_resize(stack_img, mask, 512, 512, pre_padding_len, post_padding_len)  # (h w d), (h w)

        return pil_img, img, None, msk, referring, None, datum['image']

    def _get_other_data_item(self, idx):
        raise NotImplementedError("TODO Other data is not available now.")

    def __len__(self):
        return self.epoch_quota
    
    def __getitem__(self, idx):
        
        while True:
            
            try:
                # 随机抽取一个sample
                chosen = random.choices(['deeplesion', 'radiopedia', 'Others'], weights=self.refer_seg_data_source_weight, k=1)[0]
                
                if chosen == 'deeplesion':
                    sample_idx = random.choice(self.deeplesion_idx_ls)
                    pil_img, img, pil_img_w_prmpt, msk, referring, llm_output, img_path = self._get_deeplesion_item(sample_idx)

                elif chosen == 'radiopedia':
                    sample_idx = random.choice(self.radiopedia_idx_ls)
                    pil_img, img, pil_img_w_prmpt, msk, referring, llm_output, img_path = self._get_radiopedia_item(sample_idx)
                    
                else:
                    sample_idx = random.choice(self.other_data_idx_ls)
                    pil_img, img, pil_img_w_prmpt, msk, referring, llm_output, img_path = self._get_other_data_item(sample_idx)
                
                # make sure msk is binary after rescaling
                msk = np.where(msk>0, 1.0, 0.0)
                
                # Image augmentation
                img = repeat(img, 'h w d -> c d h w', c=1)  # add channel to use monai transform
                msk = repeat(msk, 'h w -> c d h w', c=1, d=img.shape[1])
                msk = np.where(msk>0, 1.0, 0.0)
                if self.augmentator is not None:
                    data_dict = {'image': img, 'label': msk}    # 1dhw 1dhw
                    aug_data_dict = self.augmentator(data_dict)
                    img, msk = aug_data_dict['image'], aug_data_dict['label']
                img = img[0, :, :, :]       # d h w
                msk = msk[:, 0, :, :]   # 1 h w
                
                # Random drop
                for depth_idx in range(img.shape[0]):
                    if depth_idx == self.context_range:
                        continue
                    if random.random() < self.random_context_drop:
                        img[depth_idx, :, :] = 0
                    
                break

            except SystemExit:
                exit()
            except:
                # record bugs in loading data
                traceback_info = traceback.format_exc()
                print(f'*** {sample_idx} *** {chosen} ***\n')
                print(traceback_info)
            
        return {
            'task_type': 'refer_grounding_one_round',
            'vlm_data': format_one_round_data(pil_img, referring),
            'seg_img': img,
            'seg_gt': msk,
            'img_path': img_path
            }
        
if __name__ == '__main__':
    
    # Test the dataset
    dataset = Refer_Grounding_Two_Round_Dataset(
        radiopedia_json='/mnt/hwfile/medai/zhaoziheng/Med_ULS/Radiopedia_CT/data_jsonl/axial_jsonl/train(6712_axial).json',
        radiopedia_weight=1.0,
        negative_prompt_ratio=0,
        epoch_quota=1000
    )
import json
import os
from PIL import Image
from random import random
import numpy as np

from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor

from utils.distribute import main_print

PubMedVision_IMG_DIR = '/mnt/hwfile/medai/zhaoziheng/PubMedVision'

def format_one_round_data(sample):
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
                    "image": sample["image"],
                    "min_pixels": 262144,
                    "max_pixels": 262144,
                },
                {
                    "type": "text",
                    "text": sample['instruction'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["output"]
                }
            ],
        },
    ]

class Instruction_QA_Dataset(Dataset):
    
    def __init__(self, 
                 # Instruction QA Data (from PubMedVision)
                 pubmedvision_json, 
                 epoch_quota=10000, # Quota for this dataset in an epoch
                 ):
        
        self.prepare_instruction_qa_data(
            pubmedvision_json
        )
        
        self.epoch_quota = epoch_quota
        
        main_print(f"** DATA ** Total Instruction QA Samples: {len(self.data)}; Samples in One Epoch: {epoch_quota}")

    def prepare_instruction_qa_data(self, json_path):
            
        self.data = []
        
        with open(json_path, 'r') as f:
            source_data = json.load(f)
        
        for datum in source_data:
            if datum['modality'] == 'Computed Tomography' and len(datum['image']) == 1 and os.path.exists(os.path.join(PubMedVision_IMG_DIR, datum['image'][0])):
                q = datum['conversations'][0]['value']
                a = datum['conversations'][1]['value']
                self.data.append({'image':datum['image'][0], 'question':q, 'answer':a})

    def __len__(self):
        return self.epoch_quota

    def __getitem__(self, idx):
        # 随机抽取一个sample
        random_idx = np.random.randint(0, len(self.data))
        item = self.data[random_idx]
        
        # 加载原始图像（保持PIL格式）
        image_path = os.path.join(PubMedVision_IMG_DIR, item['image'])
        image = Image.open(image_path).convert('RGB')  # 确保转换为RGB
        
        sample =  {
            'image': image,  # 原始PIL图像对象
            'instruction': item['question'],  # 原始指令文本
            'output': item['answer']  # 原始输出文本
        }
        
        return {
            'task_type': 'instruction_qa',
            'vlm_data': format_one_round_data(sample)
            }
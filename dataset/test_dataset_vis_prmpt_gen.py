import json
import os
from PIL import Image

from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor

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
        }
    ]

class Vis_Prmt_Gen_Dataset(Dataset):
    
    def __init__(self, 
                 json_path, 
                 max_length=128,
                 prompt_type_ls=['bbox', 'contour', 'cropped', 'ellipse'], # include these visual propt types
                 excluded_category_json=None,   # filter these categories for external validation
                 ):
            
        if excluded_category_json is not None:
            with open(excluded_category_json, "r", encoding="utf-8") as f:
                exclude_category = json.load(f) # a dict
                exclude_category = list(exclude_category.keys())
        
        for prompt_type in prompt_type_ls:
            assert prompt_type in ['bbox', 'contour', 'cropped', 'ellipse']
            
        self.data = []
        
        with open(json_path, 'r') as f:
            source_data = json.load(f)
        
        for sample_id, content in source_data.items():
                                
            # 每一种异常信息选择每个单独分割进行prompt
            
            for abnormality_content in content['abnormality']:
                
                abnormality_id = abnormality_content['id']
                
                for prompt_type in prompt_type_ls:
                    
                    if not os.path.exists(f"{content['image']}/visual_instruction/{prompt_type}_{abnormality_id}.png"):
                        continue
                    
                    # qs = {
                    #     'cropped': 'Describe the abnormal finding at the center of this image.',
                    #     'bbox': 'Describe the abnormal finding indicated by the red box in the image.',
                    #     'contour': 'Describe the abnormal finding indicated by the red-outlined area in the image.',
                    #     'ellipse': 'Describe the abnormal finding indicated by the red circular/elliptical demarcation in the image.'
                    # }[prompt_type]
                    
                    qs = {
                        'cropped': 'What are the main findings at the center of this image? Please use precise medical terminology, maintain the concise reporting style used in formal radiology reports and provide only the specific radiological findings. Do not list general possibilities, explanations, or recommendations.',
                        'bbox': 'What are the main findings indicated by the red box in the image? Please use precise medical terminology, maintain the concise reporting style used in formal radiology reports and provide only the specific radiological findings. Do not list general possibilities, explanations, or recommendations.',
                        'contour': 'What are the main findings indicated by the red-outlined area in the image? Please use precise medical terminology, maintain the concise reporting style used in formal radiology reports and provide only the specific radiological findings. Do not list general possibilities, explanations, or recommendations.',
                        'ellipse': 'What are the main findings indicated by the red circular/elliptical demarcation in the image? Please use precise medical terminology, maintain the concise reporting style used in formal radiology reports and provide only the specific radiological findings. Do not list general possibilities, explanations, or recommendations.'
                    }[prompt_type]
                    
                    self.data.append(
                        {
                            "instruction": qs,
                            "output": abnormality_content["description"],
                            "images": [
                            f"{content['image']}/visual_instruction/{prompt_type}_{abnormality_id}.png"
                            ],
                            'id': f'{sample_id}/{abnormality_id}/{prompt_type}' 
                        }
                    )
                                
        self.max_length = max_length
        
        print(f"** DATA ** Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载原始图像（保持PIL格式）
        image_path = item['images'][0]  # 取第一个图像路径
        image = Image.open(image_path).convert('RGB')  # 确保转换为RGB
        
        sample =  {
            'image': image,  # 原始PIL图像对象
            'instruction': item['instruction'],  # 原始指令文本
            'output': item['output'],  # 原始输出文本
        }
        
        return format_one_round_data(sample), item['id']
    
class Vis_Prmt_Gen_Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples: list):
        batch_data = [example[0] for example in examples]
        batch_id = [example[1] for example in examples]
        
        # Get the texts and images, similar to training collate_fn
        texts = [self.processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in batch_data]
        image_inputs = [process_vision_info(example)[0] for example in batch_data]

        # Tokenize the texts and process the images
        batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True, truncation=True)
        
        return batch, batch_id
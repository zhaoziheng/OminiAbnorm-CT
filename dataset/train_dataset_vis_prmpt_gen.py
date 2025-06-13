import json
import os
from PIL import Image
from random import random
import numpy as np

from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor

from utils.distribute import main_print

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

class Vis_Prmt_Gen_Dataset(Dataset):
    
    def __init__(self, 
                 # Visual Prompt Data
                 json_path, 
                 rewritten_max=3,  # take GPT rewritten results as gt
                 prompt_type_ls=['bbox', 'contour', 'cropped', 'ellipse'], # include these visual propt types
                 excluded_category_json=None,   # filter these categories for external validation
                 allow_multi_prompt=True,  # whether to use multi prompt for abnormality group
                 epoch_quota=10000, # Quota for this dataset in an epoch
                 ):
        
        self.prepare_visual_prompt_data(
            json_path, 
            rewritten_max, 
            prompt_type_ls, 
            excluded_category_json, 
            allow_multi_prompt
            )
        
        self.epoch_quota = epoch_quota
        
        main_print(f"** DATA ** Total Vis_Prmt_Gen Samples: {len(self.data)}; Samples in One Epoch: {epoch_quota}")

    def prepare_visual_prompt_data(self, json_path, rewritten_max, prompt_type_ls, excluded_category_json, allow_multi_prompt):
                
        for prompt_type in prompt_type_ls:
            assert prompt_type in ['bbox', 'contour', 'cropped', 'ellipse']
            
        self.data = []
        group_prompt = 0
        single_prompt = 0
        all_prompt = 0
        
        with open(json_path, 'r') as f:
            source_data = json.load(f)
        
        for sample_id, content in source_data.items():
            
            if 'abnormality' not in content:
                continue
            
            # 合并同一张图上相同的异常信息
            
            if 'abnormality_group' in content and allow_multi_prompt:
            
                group2abnormality = {}
                for abnormality_content in content['abnormality']:
                    
                    if abnormality_content['group_id'] not in group2abnormality:
                        group2abnormality[abnormality_content['group_id']] = {
                            "id": [abnormality_content['id']],
                            "label": abnormality_content['label'],
                            "description": abnormality_content['description'],
                            "rewritten_findings": abnormality_content['rewritten_findings'] if 'rewritten_findings' in abnormality_content else [],
                        }
                    else:
                        group2abnormality[abnormality_content['group_id']]["id"].append(abnormality_content['id'])
                        if len(abnormality_content['label']) > len(group2abnormality[abnormality_content['group_id']]["label"]):
                            group2abnormality[abnormality_content['group_id']]["label"] = abnormality_content['label']
                        if len(abnormality_content['description']) > len(group2abnormality[abnormality_content['group_id']]["description"]):
                            group2abnormality[abnormality_content['group_id']]["description"] = abnormality_content['description']
                        if 'rewritten_findings' in abnormality_content:
                            group2abnormality[abnormality_content['group_id']]["rewritten_findings"].extend(abnormality_content['rewritten_findings'])
                
                # 每一种异常信息选择全部分割进行prompt
                
                for abnormality_content in group2abnormality.values():
                    
                    abnormality_id = '_'.join(str(tmp) for tmp in abnormality_content["id"])   # 0 1 2 -> 0_1_2
                    
                    for prompt_type in prompt_type_ls:
                        
                        if prompt_type == 'cropped':
                            continue
                        
                        if not os.path.exists(f"{content['image']}/visual_instruction/{prompt_type}_{abnormality_id}.png"):
                            continue
                        
                        qs = {
                            'bbox': 'Describe the abnormal finding indicated by the red boxes in the image.',
                            'contour': 'Describe the abnormal finding indicated by the red-outlined areas in the image.',
                            'ellipse': 'Describe the abnormal finding indicated by the red circular/elliptical demarcations in the image.'
                        }[prompt_type]

                        self.data.append(
                            {
                                "instruction": qs,
                                "output": abnormality_content["description"],
                                "images": [
                                f"{content['image']}/visual_instruction/{prompt_type}_{abnormality_id}.png"
                                ]
                            }
                        )
                        group_prompt += 1
                        
                        if "rewritten_findings" in abnormality_content and len(abnormality_content["rewritten_findings"]) > 0:
                            for idx, rewritten in enumerate(abnormality_content["rewritten_findings"]):
                                if idx <= rewritten_max-1:
                                    self.data.append(
                                        {
                                            "instruction": qs,
                                            "output": rewritten,
                                            "images": [
                                            f"{content['image']}/visual_instruction/{prompt_type}_{abnormality_id}.png"
                                            ]
                                        }
                                    )
                                    group_prompt += 1
                                    
                # prmpt 整张图上的所有异常信息
                
                candidate_gt_ls = []    # 每个group一个sublist，sublist内是所有等价的findings
                all_abnormality_has_findings = True
                
                for abnormality_group in group2abnormality.values():
                
                    if 'description' not in abnormality_group or len(abnormality_group['description']) == 0:
                        if 'label' not in abnormality_group or len(abnormality_group['label']) == 0:
                            candidate_gt = []
                        else:
                            candidate_gt = [abnormality_group['label']]
                    else:
                        candidate_gt = [abnormality_group['description']]
                        
                    if 'rewritten_findings' in abnormality_group and len(abnormality_group['rewritten_findings']) > 0:
                        candidate_gt += abnormality_group['rewritten_findings']
                        
                    if len(candidate_gt) == 0:
                        all_abnormality_has_findings = False
                        break
                    else:
                        candidate_gt_ls.append(candidate_gt)
                    
                if all_abnormality_has_findings:    # 图上有的异常没有对应的findings，则放弃整张图像
                
                    for prompt_type in prompt_type_ls:
                        
                        if prompt_type == 'cropped':
                                continue
                        
                        if not os.path.exists(f"{content['image']}/visual_instruction/{prompt_type}_all.png"):
                            continue
                        
                        # qs = {
                        #     'bbox': 'Describe all the abnormal findings indicated by the red boxes in the image.',
                        #     'contour': 'Describe all the abnormal findings indicated by the red-outlined areas in the image.',
                        #     'ellipse': 'Describe all the abnormal findings indicated by the red circular/elliptical demarcations in the image.'
                        # }[prompt_type]
                        
                        qs = {
                            'bbox': 'What are the main findings indicated by the red boxes in the image? Please use precise medical terminology, maintain the concise reporting style used in formal radiology reports and provide only the specific radiological findings. Do not list general possibilities, explanations, or recommendations.',
                            'contour': 'What are the main findings indicated by the red-outlined areas in the image? Please use precise medical terminology, maintain the concise reporting style used in formal radiology reports and provide only the specific radiological findings. Do not list general possibilities, explanations, or recommendations.',
                            'ellipse': 'What are the main findings indicated by the red circular/elliptical demarcations in the image? Please use precise medical terminology, maintain the concise reporting style used in formal radiology reports and provide only the specific radiological findings. Do not list general possibilities, explanations, or recommendations.'
                        }[prompt_type]

                        for _ in range(4):
                            # Create a complete GT by randomly selecting one element from each list
                            answer = ""
                            for i, candidate_list in enumerate(candidate_gt_ls):
                                gt = np.random.choice(candidate_list)
                                answer += f'Findings {i+1}: {gt}\n'
                            
                            # Add this complete GT to the dataset
                            self.data.append({
                                "instruction": qs,
                                "output": answer,
                                "images": [f"{content['image']}/visual_instruction/{prompt_type}_all.png"]
                            })

                        all_prompt += 1
                                
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
                            ]
                        }
                    )
                    single_prompt += 1
                    
                    if "rewritten_findings" in abnormality_content and len(abnormality_content["rewritten_findings"]) > 0:
                        for idx, rewritten in enumerate(abnormality_content["rewritten_findings"]):
                            if idx <= rewritten_max-1:
                                self.data.append(
                                    {
                                        "instruction": qs,
                                        "output": rewritten,
                                        "images": [
                                        f"{content['image']}/visual_instruction/{prompt_type}_{abnormality_id}.png"
                                        ]
                                    }
                                )
                                single_prompt += 1

        # 统计数据集的样本数量
            
        main_print(f'** DATA ** Vis_Prmpt_Gen Task has {all_prompt} all prompt samples, {group_prompt} group prompt samples, {single_prompt} single prompt samples.')

    def _generate_instruction(self, prompt_type):
        # 共用的前缀
        prefixes_punctuations = [
            ("Describe", "."),
            ("Please describe", "."),
            ("Can you describe",  "?"),
            ("Provide a description of", "."),
            ("Characterize", "."),
            ("Please characterize", "."),
            ("Can you characterize",  "?"),
            ("Explain", "."),
            ("Please explain", "."),
            ("Can you explain",  "?"),
            ("Elaborate on", "."),
            ("Please elaborate on", "."),
            ("Can you elaborate on",  "?"),
            ("Analyze", "."),
            ("Please analyze", "."),
            ("Can you analyze",  "?"),
            ("Detail" "."),
            ("Please detail", "."),
            ("Can you detail",  "?"),
            ("Give details about", "."),
        ]
        
        # 共用的连接词
        connections = [
            "in the image", 
            "on the image",
            "in this image", 
            "on this image",
            "in the CT image", 
            "on the CT image",
            "in this CT image", 
            "on this CT image",
        ]
        
        pronouns = ['abnormal finding', 'abnormality', 'finding']
        
        indication_verbs = [
            "indicated", 
            "marked", 
            "highlighted", 
            "outlined",
            "identified",
        ]
        
        # 根据提示类型选择合适的描述词
        if prompt_type == 'cropped':
            descriptors = [
                "at the center of this image",
                "located in the center of the image",
                "present in the center of the image",
            ]
            # 裁剪类型特殊处理，不需要连接词
            template = "{prefix} the {pronoun} {descriptor}{punctuation}"
        else:
            if prompt_type == 'bbox':
                descriptors = [
                    "red box", 
                    "red bounding box", 
                    "red-outlined box", 
                    "highlighted red area",
                    "red rectangular region"
                ]
            elif prompt_type == 'contour':
                descriptors = [
                    "red-outlined area", 
                    "area with the red contour", 
                    "region outlined in red", 
                    "red contoured region", 
                    "red-bordered region"
                ]
            elif prompt_type == 'ellipse':
                descriptors = [
                    "red circle/ellipse",
                    "red circular/elliptical demarcation", 
                    "red circular/elliptical marking", 
                    "red circular/elliptical outline", 
                ]
            template = "{prefix} the {pronoun} {indication_verb} by the {descriptor} {connection}{punctuation}"
        
        # 随机选择各个部分
        prefix, punctuation = random.choice(prefixes_punctuations)
        pronoun = random.choice(pronouns)
        indication_verb = random.choice(indication_verbs)
        descriptor = random.choice(descriptors)
        connection = random.choice(connections)
        
        # 根据模板生成指令
        if prompt_type == 'cropped':
            return template.format(prefix=prefix, pronoun=pronoun, descriptor=descriptor, punctuation=punctuation)
        else:
            return template.format(prefix=prefix, pronoun=pronoun, indication_verb=indication_verb, descriptor=descriptor, connection=connection, punctuation=punctuation)

    def __len__(self):
        return self.epoch_quota

    def __getitem__(self, idx):
        # 随机抽取一个sample
        random_idx = np.random.randint(0, len(self.data))
        item = self.data[random_idx]
        
        # 加载原始图像（保持PIL格式）
        image_path = item['images'][0]  # 取第一个图像路径
        image = Image.open(image_path).convert('RGB')  # 确保转换为RGB
        
        sample =  {
            'image': image,  # 原始PIL图像对象
            'instruction': item['instruction'],  # 原始指令文本
            'output': item['output']  # 原始输出文本
        }
        
        return {
            'task_type': 'vis_prmpt_gen',
            'vlm_data': format_one_round_data(sample)
            }
    
if __name__ == '__main__':
    
    from tqdm import tqdm
    
    train_dataset = Vis_Prmt_Gen_Dataset('/mnt/hwfile/medai/zhaoziheng/Med_ULS/Radiopedia_CT/data_jsonl/mixed_jsonl/merged_train(10993_mixed).json')
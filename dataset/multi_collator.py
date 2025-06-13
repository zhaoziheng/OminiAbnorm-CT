from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor
import torch

class MultiTypeCollator:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, samples: list):
        task_type_of_this_batch = [t['task_type'] for t in samples]
        if len(set(task_type_of_this_batch)) > 1:
            raise ValueError(f"All samples in a batch must have the same task_type. Found: {set(task_type_of_this_batch)}")
        
        chat_format_data = [t['vlm_data'] for t in samples]
                
        # Get the texts and images, and apply the chat template
        texts = [self.processor.apply_chat_template(datum, tokenize=False) for datum in chat_format_data]  # Prepare texts for processing
        image_inputs = [process_vision_info(datum)[0] for datum in chat_format_data]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  # Encode texts and images into tensors
        
        labels = torch.full_like(batch["input_ids"], -100)  # Initialize labels with all -100 values
    
        # Find where assistant tokens begin and end in each example
        for i in range(len(texts)):
            # Get the positions where the assistant's responses begin and end
            input_ids = batch["input_ids"][i].tolist()
            
            # Look for the assistant role markers in the tokenized input
            assistant_start_tokens = self.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            assistant_end_tokens = self.processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            
            # Initialize pointer
            idx = 0
            while idx < len(input_ids):
                # Find next start marker
                start_pos = None
                for s_idx in range(idx, len(input_ids) - len(assistant_start_tokens) + 1):
                    if input_ids[s_idx:s_idx+len(assistant_start_tokens)] == assistant_start_tokens:
                        start_pos = s_idx + len(assistant_start_tokens)
                        break
                
                if start_pos is None:
                    # No more start markers
                    break
                
                # Find corresponding end marker
                end_pos = None
                for e_idx in range(start_pos, len(input_ids) - len(assistant_end_tokens) + 1):
                    if input_ids[e_idx:e_idx+len(assistant_end_tokens)] == assistant_end_tokens:
                        end_pos = e_idx
                        break
                
                if end_pos is None:
                    # No end marker found, break
                    break
                
                # Unmask tokens between start(excluded) and end(included) (keep these labels)
                labels[i, start_pos:end_pos+1] = batch["input_ids"][i, start_pos:end_pos+1].clone()
                
                # Continue search after this end token
                idx = end_pos + len(assistant_end_tokens)

        batch["labels"] = labels  # Add labels to the batch
        
        # batch input for segmentation model
        segmentation_image = []
        segmentation_gt_mask = []
        for t in samples:
            if 'seg_img' in t:
                if isinstance(t['seg_img'], torch.Tensor):
                    segmentation_image.append(t['seg_img'])
                else:
                    segmentation_image.append(torch.tensor(t['seg_img']))
                if isinstance(t['seg_gt'], torch.Tensor):
                    segmentation_gt_mask.append(t['seg_gt'])
                else:
                    segmentation_gt_mask.append(torch.tensor(t['seg_gt']))
            else:
                segmentation_image.append(torch.zeros((9, 512, 512), dtype=torch.float32)) 
                segmentation_gt_mask.append(torch.zeros((1, 512, 512), dtype=torch.float32))
                
        batch['segmentation_image'] = torch.stack(segmentation_image, dim=0)  # b d h w
        batch['segmentation_gt_mask'] = torch.stack(segmentation_gt_mask)   # b 1 h w
        
        return batch  # Return the prepared batch
    
    
import torch
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
from einops import repeat

from qwen_vl_utils import process_vision_info

from utils.online_prmpt_generation import draw_bounding_box, generate_second_round_input
from dataset.test_dataset_grounding_gen import format_data
    
def save_segmentation_mask(image_slice, prediction, transform_info, save_path):
    """
    Save segmentation mask to file.
    
    Args:
        image_slice: 2D numpy array (H, W) - the original image slice for reference
        prediction: Segmentation mask as numpy array (512, 512)
        transform_info: Transformation metadata from pad_truncate_and_resize
        save_path: Path to save the mask
        original_image_shape: Original image shape for resizing if needed
    """
    # Extract the mask (remove batch and channel dimensions)
    recovered_mask = recover_single_slice(prediction, transform_info)
    
    # Convert to 0-255 range
    mask_uint8 = (recovered_mask * 255).astype(np.uint8)
    image_slice = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice)) * 255.0
    
    # Convert to RGB
    rgb_img = repeat(image_slice, 'h w -> h w c', c=3)
    rgb_pred = repeat(mask_uint8, 'h w -> h w c', c=3)
    rgb_pred[:, :, 1:] = 0   # paint with R
    
    overlap_pred = rgb_img * 0.3 + rgb_pred * 0.7
    
    # Save as image
    cv2.imwrite(save_path, overlap_pred)
    print(f"Segmentation mask saved to: {save_path}")
    
def get_pil_image(img):
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

def pad_truncate_and_resize(stack_img, target_h, target_w, pre_padding_len, post_padding_len):
    """
    pad image to H W D and return transformation info for recovery
    """
    original_h, original_w, original_d = stack_img.shape
    
    # Store transformation metadata
    transform_info = {
        'original_shape': (original_h, original_w, original_d),
        'target_shape': (target_h, target_w),
        'pre_padding_len': pre_padding_len,
        'post_padding_len': post_padding_len,
        'actual_pre_truncate': max(0, -pre_padding_len),
        'actual_post_truncate': max(0, -post_padding_len),
        'hw_padding': None  # Will be set below
    }
    
    h, w, d = stack_img.shape
    
    padded_img = []
    
    # depth padding or truncate
    if pre_padding_len > 0:
        padded_img.append(np.zeros(shape=(h, w, pre_padding_len)))
    if pre_padding_len < 0:
        stack_img = stack_img[:, :, (-1)*pre_padding_len:]
        
    padded_img.append(stack_img)
    
    if post_padding_len > 0:
        padded_img.append(np.zeros(shape=(h, w, post_padding_len)))
        
    padded_img = np.concatenate(padded_img, axis=-1)
    
    if post_padding_len < 0:
        padded_img = padded_img[:, :, :post_padding_len]
    
    # h w padding
    if h > w:
        left_padding_len = (h - w) // 2
        right_padding_len = h - w - left_padding_len
        padded_img = np.pad(padded_img, ((0, 0), (left_padding_len, right_padding_len), (0, 0)), 'constant')
        transform_info['hw_padding'] = ('width', left_padding_len, right_padding_len)
    elif w > h:
        up_padding_len = (w - h) // 2
        down_padding_len = w - h - up_padding_len
        padded_img = np.pad(padded_img, ((up_padding_len, down_padding_len), (0, 0), (0, 0)), 'constant')
        transform_info['hw_padding'] = ('height', up_padding_len, down_padding_len)
    else:
        transform_info['hw_padding'] = ('none', 0, 0)

    # Resize each slice of the 3D image stack
    resized_slices = []
    for d_idx in range(padded_img.shape[2]):
        slice_2d = padded_img[:, :, d_idx]
        resized_slice = cv2.resize(slice_2d, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_slices.append(resized_slice)
    resized_img = np.stack(resized_slices, axis=2)
    
    return resized_img, transform_info

def recover_single_slice(processed_slice_2d, transform_info):
    """
    Recover a single 2D slice from processed image using transformation info
    
    Args:
        processed_slice_2d: 2D array (h, w) - single slice from processed 3D image
        slice_idx: index of the slice in the processed image
        transform_info: transformation metadata from pad_truncate_and_resize
    """
    original_h, original_w, original_d = transform_info['original_shape']
    
    # Step 1: Reverse resize - resize back to padded dimensions
    padded_h = max(original_h, original_w)
    padded_w = max(original_h, original_w)
    
    recovered_slice = cv2.resize(processed_slice_2d, (padded_w, padded_h), interpolation=cv2.INTER_LINEAR)
    
    # Step 2: Remove H/W padding
    hw_padding_type, pad1, pad2 = transform_info['hw_padding']
    if hw_padding_type == 'width':  # original h > w, remove width padding
        recovered_slice = recovered_slice[:, pad1:padded_w-pad2]
    elif hw_padding_type == 'height':  # original w > h, remove height padding
        recovered_slice = recovered_slice[pad1:padded_h-pad2, :]
    
    return recovered_slice

def preprocess_input(img_dir, raw_text_instruction, processor):
    """ 
    Preprocess the input image(s) and instruction text for the model.

    Args:
        img_dir (str): Path to image directory or single image file
        raw_text_instruction (str): Text instruction for the model
        processor: Model processor for tokenization and image processing

    Returns:
        dict: Dictionary containing preprocessed data with keys:
            - vllm_input: Tokenized input for the VLM model
            - vllm_raw_input: Raw input data for VLM
            - seg_img: Preprocessed image tensor for segmentation
            - transform_info: Transformation metadata to recover original shape
            - center_img: Original center image slice (w/o padding or resize)
    """
    
    if os.path.isdir(img_dir):
    
        # Get all image files in the directory
        image_files = []
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract the number from filename (e.g., "0.png" -> 0)
                basename = os.path.splitext(os.path.basename(filename))[0]
                if not basename.isdigit():
                    continue
                name_without_ext = os.path.splitext(filename)[0]
                image_files.append((int(name_without_ext), filename))

        # Sort by the numeric part
        image_files.sort(key=lambda x: x[0])

        # Read images in order and store in list
        img_list = []
        for _, filename in image_files:
            img_path = os.path.join(img_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = (img - np.mean(img)) / (np.std(img) + 1e-10)
                img_list.append(img)

        # Find the middle image (for odd length, true middle; for even length, second middle)
        assert len(img_list) > 0, f'No images found under {img_dir}'
        center_index = len(img_list) // 2
        center_img = img_list[center_index]
        # Resize all images to match center_img shape
        for i in range(len(img_list)):
            if img_list[i].shape != center_img.shape:
                img_list[i] = cv2.resize(img_list[i], (center_img.shape[1], center_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                print(f'Resize image {i} to center image shape {center_img.shape[1], center_img.shape[0]}')
        stack_img = np.stack(img_list, axis=-1)
        
    elif img_dir.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        center_img = img = (img - np.mean(img)) / (np.std(img) + 1e-10)
        center_index = 0
        stack_img = np.expand_dims(img, axis=-1)  # h w 1
        
    else:
        raise ValueError(f"Unsupported image input: {img_dir}. Please provide a valid image directory or file.")
    
    # Find images before and after center image
    pre_len = center_index
    post_len = stack_img.shape[-1] - center_index - 1
    context_range = 4
    pre_padding_len = max(0, context_range - pre_len)
    post_padding_len = max(0, context_range - post_len)

    # padding and resize
    seg_img, transform_info = pad_truncate_and_resize(stack_img, 512, 512, pre_padding_len, post_padding_len)
    seg_img = repeat(seg_img, 'h w d -> b d h w', b=1)
    seg_img = torch.tensor(seg_img)
    
    # load pil image
    pil_img = get_pil_image(center_img)
    
    # formulate input
    vllm_raw_input = [format_data(pil_img, raw_text_instruction)]
    
    # process
    texts = [processor.apply_chat_template(vllm_raw_input[0], tokenize=False, add_generation_prompt=True)]
    image_inputs = [process_vision_info(vllm_raw_input[0])[0]]
    
    # Tokenize the texts and process the images
    vllm_input = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True, truncation=True)
    
    return {
        'vllm_input': vllm_input,
        'vllm_raw_input': vllm_raw_input,
        'seg_img': seg_img,
        'transform_info': transform_info,   # to recover prediction mask
        'center_img': center_img    # unresized 
    }
    
def load_model(base_model_path, adapter_path, tokenizer_path, seg_lm_head_path):
    
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, AutoTokenizer

    model_path = base_model_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    # Load LoRA weights

    model.load_adapter(adapter_path)

    # Load or modify processor
    
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path, padding_side="left", use_fast=True)
    
    if tokenizer_path is not None:
        processor.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="left"
        )
    else:
        special_tokens_dict = {'additional_special_tokens': ['<SEG>']}
        processor.tokenizer.add_special_tokens(special_tokens_dict)
    
    # Expand the vocabulary with <SEG> 
        
    model.resize_token_embeddings(len(processor.tokenizer))
    seg_token_idx = processor.tokenizer.convert_tokens_to_ids('<SEG>')
    
    # Load lm head, embedding
    
    other_trainable_param_ckpt = torch.load(seg_lm_head_path)
    
    loaded_param = []
    missed_param = []
    for n, p in model.named_parameters():
        if "lm_head" in n:
            if 'lm_head_state_dict' in other_trainable_param_ckpt and n in other_trainable_param_ckpt['lm_head_state_dict']:
                loaded_param.append(n)
                p.data = other_trainable_param_ckpt['lm_head_state_dict'][n].data
            else:
                missed_param.append(n)
            p.requires_grad = True
        if "embed_tokens" in n:
            if 'embed_tokens_state_dict' in other_trainable_param_ckpt and n in other_trainable_param_ckpt['embed_tokens_state_dict']:
                loaded_param.append(n)
                p.data = other_trainable_param_ckpt['embed_tokens_state_dict'][n].data
            else:
                missed_param.append(n)
            p.requires_grad = True
            
    # To GPU
    model = model.cuda()
    model.eval()
    model = torch.compile(model)
    
    # Load seg model

    from model.unet import UNET
        
    seg_model = UNET(input_channels=9)
    seg_model = seg_model.cuda()
    seg_model.load_saved(other_trainable_param_ckpt)
    seg_model.eval()
    
    return model, seg_model, processor, seg_token_idx

def generate_text_and_grounding(
    model, 
    seg_model,
    processor,
    seg_token_idx,
    input_data
):
    first_round_vllm_input = input_data['vllm_input']
    vllm_raw_input = input_data['vllm_raw_input']
    seg_img = input_data['seg_img'].cuda()

    first_round_vllm_input = {k: v.cuda() for k, v in first_round_vllm_input.items()}
    
    # Generate text with the vlm
    with torch.no_grad():
        vlm_outputs = model.generate(
            **first_round_vllm_input,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=1024
            )
        
        generated_ids = vlm_outputs.sequences
        
        # Trim the generated ids to remove the input ids
        trimmed_generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(first_round_vllm_input["input_ids"], generated_ids)
        ]
        
        # Decode the output texts
        first_round_output_texts = processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )[0]

        if '<SEG>' not in first_round_output_texts:
            # Remove all special tokens except <SEG>
            special_tokens_to_remove = [token for token in processor.tokenizer.all_special_tokens]
            for token in special_tokens_to_remove:
                first_round_output_texts = first_round_output_texts.replace(token, '')
            
            output_texts = first_round_output_texts
            seg_pred = None
            
        else:
            # invoke Segmentation model
            hidden_states = vlm_outputs.hidden_states
            last_layer_hidden_states = [state[-1] for state in hidden_states]
            final_hidden_state = torch.cat(last_layer_hidden_states, dim=1)

            # For each sequence in the batch try to find the 1st seg token and its hidden state embedding
            seg_token_mask = generated_ids == seg_token_idx
            seg_positions = torch.where(seg_token_mask[0])[0]
            seg_latent_embeddings = final_hidden_state[0, seg_positions[0]]
            seg_latent_embeddings = repeat(seg_latent_embeddings, 'd -> b n d', b=1, n=1)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                seg_img = seg_img.to(torch.bfloat16)
                prediction = seg_model(seg_img, seg_latent_embeddings) # (1, 1, H, W)
                prediction = torch.sigmoid(prediction)
                prediction = torch.where(prediction>0.5, 1.0, 0.0)
                seg_pred = prediction.detach().cpu().numpy().squeeze()  # (H, W)
            
            # from seg mask to vis_prmpt_img
            center_slice_idx = seg_img.shape[1] // 2
            image_to_draw_on = seg_img[0, center_slice_idx, ...] # h w
            image_np = image_to_draw_on.float().detach().cpu().numpy()
            image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-8)
            image_np = (image_np * 255).astype(np.uint8)
            vis_prmpt_img = draw_bounding_box(image_np, seg_pred)
            vis_prmpt_img = cv2.cvtColor(vis_prmpt_img, cv2.COLOR_BGR2RGB)
            vis_prmpt_img = Image.fromarray(vis_prmpt_img)
            
            second_round_input = generate_second_round_input(processor, vllm_raw_input, [first_round_output_texts], [vis_prmpt_img])
            second_round_input = {k: v.cuda() for k, v in second_round_input.items()}

            second_round_outputs = model.generate(
                **second_round_input,
                max_new_tokens=1024
            )
            
            # Trim the generated ids to remove the input ids
            second_round_trimmed_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(second_round_input["input_ids"], second_round_outputs)
            ]
            
            # Decode the output texts from the second round
            second_round_texts = processor.batch_decode(
                second_round_trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Update the output texts with the second round results
            output_texts = second_round_texts[0]
                
        return output_texts, seg_pred

def chat(image_path, text_input, output_path=None):
    
    if os.path.isdir(image_path):
        output_path = os.path.join(image_path, "prediction.jpeg")
        
    elif image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_prediction.jpeg"
    
    # print("Loading model...")
    # Load model
    model, seg_model, processor, seg_token_idx = load_model(
        '/mnt/hwfile/medai/LLMModels/Model/Qwen2.5-VL-7B-Instruct', 
        './checkpoints/ominiabnorm-ct-7b', 
        './checkpoints/ominiabnorm-ct-7b/extended_tokenizer', 
        './checkpoints/ominiabnorm-ct-7b/seg_model_lm_head_embed_tokens.pt'
    )
    
    # print("Processing input...")
    # Process input
    processed_input = preprocess_input(
        image_path, 
        text_input, 
        processor
    )
    
    # print("Generating analysis...")
    # Generate output
    output_texts, seg_pred = generate_text_and_grounding(
        model, 
        seg_model,
        processor, 
        seg_token_idx, 
        processed_input
    )
    
    # Save results
    if seg_pred is not None:
        save_segmentation_mask(
            processed_input['center_img'], 
            seg_pred, 
            processed_input["transform_info"], 
            output_path
        )
        print(f"Grounding result saved to: {output_path}")
    
    print(f"Answer:\n{output_texts}")

if __name__ == "__main__":
    
    # # example usage 1
    # image_path = "/mnt/petrelfs/zhaoziheng/Med-ULS/OminiAbnorm-CT/demo/demo1"
    # text_input = "Please localize all the abnormalities on this CT image and describe the findings."
    # chat(image_path, text_input)
    
    # # example usage 2
    # image_path = "/mnt/petrelfs/zhaoziheng/Med-ULS/OminiAbnorm-CT/demo/demo2"
    # text_input = "The patient is experiencing pelvic pain and chronic constipation. Evaluate the uterus for any abnormalities such as enlargement or possible masses that could be linked to these symptoms. If any, localize it on the image and describe the findings."
    # chat(image_path, text_input)
    
    # # example usage 3
    # image_path = "/mnt/petrelfs/zhaoziheng/Med-ULS/OminiAbnorm-CT/demo/demo3/demo3.jpeg"
    # text_input = 'What are the main findings indicated by the red box in the image? Please use precise medical terminology, maintain the concise reporting style used in formal radiology reports and provide only the specific radiological findings. Do not list general possibilities, explanations, or recommendations.'
    # chat(image_path, text_input)
    
    # Get user input
    image_path = input("Please enter the path to the CT image or an image directory: ").strip()
    text_input = input("Please enter your text instruction: ").strip()
    chat(image_path, text_input)
import torch
import numpy as np
from einops import repeat, rearrange, reduce
import argparse
import os
import time
import pandas as pd
import pickle
import json
from pathlib import Path
import cv2
from PIL import Image

from utils.online_prmpt_generation import draw_bounding_box, generate_second_round_input

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    
    # Do this if you encounter annoying warnings from torchvision transform
    
    import warnings
    warnings.filterwarnings("ignore", message="The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17")
    warnings.filterwarnings("ignore", message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta.")
    
    # Parse configuration command line
    
    parser = argparse.ArgumentParser(description='Evaluation script for grounding task')
    # DATA
    parser.add_argument('--deeplesion_json', type=str, default=None)
    parser.add_argument('--radiopedia_json', type=str, default=None)
    parser.add_argument('--other_data_json', type=str, default=None)
    parser.add_argument('--dataset_config_dict', type=str, default='./dataset/config/other_datasets.json')
    # EXP
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--rcd_dir', type=str, required=True, help='Save the evaluation results (in a directory)')
    parser.add_argument('--rcd_file', type=str, help='Save the grounding results (in a csv/xlsx file)')
    parser.add_argument('--visualization', type=str2bool, default=False, help='Batch size for inference')
    # MODEL
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to the LoRA adapter weights')
    parser.add_argument('--tokenizer_path', type=str, help='Path to a modified tokenizer (with <SEG> added as special token)')
    parser.add_argument('--other_trainable_param_path', type=str, required=True, help='To load seg model, lm_head, embed_tokens (all the orther trainable param beside lora layers).')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference (per device)')
    args = parser.parse_args()

    # Initialize distributed environment
    
    from utils.distribute import main_print, is_main_process, get_rank, get_world_size
    
    if 'RANK' in os.environ:
        is_distributed = True
        torch.distributed.init_process_group(backend="nccl")
        local_rank = get_rank()
        torch.cuda.set_device(local_rank)
        main_print(f"** DDP ** Rank: {local_rank}, World Size: {get_world_size()}")
    else:
        is_distributed = False  # WARNING May not work properly without ddp
        
    # Path
    
    if args.rcd_file is None:
        args.rcd_file = 'ground_gen_' + os.path.basename(args.radiopedia_json).replace('.json', '')
    
    Path(args.rcd_dir).mkdir(parents=True, exist_ok=True)
    csv_path = f'{args.rcd_dir}/(seg){args.rcd_file}.csv'
    visual_dir = csv_path.replace('.csv', '')
    if args.visualization:
        Path(visual_dir).mkdir(parents=True, exist_ok=True)
    main_print(f'** EXP ** Saved Seg Results to {csv_path}')
    
    # Model

    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, AutoTokenizer

    model_path = args.model_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    # Load LoRA weights

    model.load_adapter(args.adapter_path)

    # Load or modify processor
    
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path, padding_side="left", use_fast=True)
    
    if args.tokenizer_path is not None:
        processor.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            padding_side="left"
        )
    else:
        special_tokens_dict = {'additional_special_tokens': ['<SEG>']}
        processor.tokenizer.add_special_tokens(special_tokens_dict)
    
    # Expand the vocabulary with <SEG> 
        
    model.resize_token_embeddings(len(processor.tokenizer))
    seg_token_idx = processor.tokenizer.convert_tokens_to_ids('<SEG>')
    
    # Load lm head, embedding
    
    main_print(f"** MODEL ** Try to loading some weights from {args.other_trainable_param_path}")
    other_trainable_param_ckpt = torch.load(args.other_trainable_param_path)
    
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
    main_print(f"** VLM MODEL ** Parameters not in checkpoint : {missed_param}")
    main_print(f"** VLM MODEL ** Parameters loaded from checkpoint : {loaded_param}")
    
    # Load seg model

    from model.unet import UNET
        
    seg_model = UNET(input_channels=9)
    seg_model = seg_model.to(get_rank())
    seg_model.load_saved(other_trainable_param_ckpt)
    seg_model.eval()

    # Dataset
    
    from dataset.test_dataset_grounding_gen import Grounding_Gen_Dataset, Grounding_Gen_Collator
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    test_dataset = Grounding_Gen_Dataset(
        deeplesion_json = args.deeplesion_json,
        radiopedia_json = args.radiopedia_json,
        other_data_json = args.other_data_json,
        dataset_config_dict = args.dataset_config_dict
        )
    
    collator = Grounding_Gen_Collator(processor)

    # Set up distributed sampler if using distributed
    
    if is_distributed:
        sampler = DistributedSampler(test_dataset, shuffle=False)
        sampler.set_epoch(0)
    else:
        sampler = None

    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False if sampler else False,
        sampler=sampler,
        collate_fn=collator,
        num_workers=16,
    )
    
    # To GPU
    model = model.to(get_rank())
    
    # Set to evaluation mode
    model.eval()
        
    # Compile
    model = torch.compile(model)

    # Inference Interface

    import tqdm
    
    def generate_text_and_grounding_from_batch(model, batch, max_new_tokens=1024, device="cuda"):
        # unpack
        first_round_vllm_input = batch['vllm_input']
        first_round_vllm_raw_input = batch['vllm_raw_input']
        images = batch['seg_img'].to(device)  # b d h w
        gt_segmentation = batch['seg_gt'].numpy() # b 1 h w   
        batch_size = images.size(0)

        # Move batch to device
        first_round_vllm_input = {k: v.to(device) for k, v in first_round_vllm_input.items()}
        
        # Generate text with the vlm
        with torch.no_grad():
            vlm_outputs = model.generate(**first_round_vllm_input,
                                         output_scores=True,
                                         output_hidden_states=True,
                                         return_dict_in_generate=True,
                                         max_new_tokens=max_new_tokens)

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
            )

            # Remove all special tokens except <SEG>
            special_tokens_to_remove = [token for token in processor.tokenizer.all_special_tokens if token != '<SEG>']
            for i in range(len(first_round_output_texts)):
                for token in special_tokens_to_remove:
                    first_round_output_texts[i] = first_round_output_texts[i].replace(token, '')

            hidden_states = vlm_outputs.hidden_states
            last_layer_hidden_states = [state[-1] for state in hidden_states]
            final_hidden_state = torch.cat(last_layer_hidden_states, dim=1)
            hidden_dim = final_hidden_state.size(-1)

            max_seg_tokens = 1  # We'll only use the first SEG token if there are multiple
            seg_latent_embeddings = torch.zeros(batch_size, max_seg_tokens, hidden_dim, device=device)
            query_mask = torch.zeros(batch_size, max_seg_tokens, device=device)

            # For each sequence in the batch try to find the 1st seg token and its hidden state embedding
            seg_token_mask = generated_ids == seg_token_idx
            for b in range(batch_size):
                seg_positions = torch.where(seg_token_mask[b])[0]
                if len(seg_positions) > 0:
                    seg_latent_embeddings[b, 0] = final_hidden_state[b, seg_positions[0]]
                    query_mask[b, 0] = 1
            
            prediction = np.zeros_like(gt_segmentation)
            output_texts = first_round_output_texts
            
            # Call seg model if any sample in the batch trigger <SEG>
            if query_mask.sum().item() > 0:
                        
                # Now seg_latent_embeddings has shape (batch_size, max_token_num, hidden_state_dim)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    images = images.to(torch.bfloat16)
                    prediction = seg_model(images, seg_latent_embeddings) # (B, max_seg_tokens, H, W)
                    prediction = torch.sigmoid(prediction)
                    prediction = torch.where(prediction>0.5, 1.0, 0.0)
                    # Maskout padded tokens (to zeros)
                    H, W = prediction.shape[-2:]
                    expanded_query_mask = query_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
                    prediction = prediction * expanded_query_mask
                    prediction = prediction.detach().cpu().numpy()
                    
                # from seg mask to vis_prmpt_img
                vis_prmpt_img_ls = []
                for b in range(batch_size):
                    # find the center image and convert to 0～255 numpy array
                    center_slice_idx = images.shape[1] // 2
                    image_to_draw_on = images[b, center_slice_idx, ...] # h w
                    image_np = image_to_draw_on.float().detach().cpu().numpy()
                    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-8)
                    image_np = (image_np * 255).astype(np.uint8)
                    vis_prmpt_img = draw_bounding_box(image_np, prediction[b, 0, ...])
                    vis_prmpt_img = cv2.cvtColor(vis_prmpt_img, cv2.COLOR_BGR2RGB)
                    vis_prmpt_img = Image.fromarray(vis_prmpt_img)
                    vis_prmpt_img_ls.append(vis_prmpt_img)
                    
                # Prepare new prompts with the segmentation visualization
                batch_second_round_input = generate_second_round_input(processor, first_round_vllm_raw_input, output_texts, vis_prmpt_img_ls)
                batch_second_round_input = {k: v.to(device) for k, v in batch_second_round_input.items()}
                
                # Generate text with the vlm for the second round
                with torch.no_grad():
                    second_round_outputs = model.generate(
                        **batch_second_round_input,
                        max_new_tokens=max_new_tokens
                    )
                    
                    # Trim the generated ids to remove the input ids
                    second_round_trimmed_ids = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(batch_second_round_input["input_ids"], second_round_outputs)
                    ]
                    
                    # Decode the output texts from the second round
                    second_round_texts = processor.batch_decode(
                        second_round_trimmed_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    
                    # Update the output texts with the second round results
                    output_texts = second_round_texts
                
            # for sample in batch fail to trigger <SEG> sucessfully
            for b in range(batch_size):
                if query_mask[b, 0] == 0:
                    # ensure their dice to 0
                    if gt_segmentation[b, ...].sum() == 0:
                        prediction[b, ...] = np.ones_like(gt_segmentation)
                    elif gt_segmentation[b, ...].sum() > 0:
                        prediction[b, ...] = np.zeros_like(gt_segmentation)
                    # first round output as output_texts
                    output_texts[b] = first_round_output_texts[b]
                    
        return output_texts, prediction 
    
    # Try to load previous evaluation mediate results

    # 1. find all tmp and fnl pickle files
    # 2. load the results, merge into tmp_rank_0
    # 3. find their ids, and avoid re-evaluating them
    evaluated_samples = set()
    results_of_samples = []
    if args.resume:
        if is_main_process():
            loaded_files = []
        tmp_prefix = os.path.basename(csv_path).replace('.csv', '_tmp_rank')  # xxx/test/step_xxx.csv --> step_xxx_tmp_rank
        fnl_prefix = os.path.basename(csv_path).replace('.csv', '_fnl_rank')  # xxx/test/step_xxx.csv --> step_xxx_fnl_rank
        for file_name in os.listdir(args.rcd_dir):
            if tmp_prefix in file_name or fnl_prefix in file_name:
                # load list of results
                with open(f'{args.rcd_dir}/{file_name}', 'rb') as f:
                    tmp = pickle.load(f)    
                    if is_main_process():
                        loaded_files.append(f'{args.rcd_dir}/{file_name}')
                        results_of_samples += tmp
                for line in tmp:    # each line : [dataset_name, sample_id, ... ...] 
                    evaluated_samples.add(f'{line[0]}_{line[1]}')
        
        # NOTE: all the process should aware the ids of evaluated samples, BUT only the master need to load all the results, and delete previous pickle            
        if is_main_process():
            for file_path in loaded_files:
                os.remove(file_path)
                print(f'Load and Remove {file_path}')
            merge_pkl = csv_path.replace('.csv', '_tmp_rank0.pkl')
            with open(merge_pkl, 'wb') as f:
                pickle.dump(results_of_samples, f)
                print(f'Load results of {len(results_of_samples)} samples, Merge into {merge_pkl}')
    
    # Evaluate in scale
    
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from evaluator.grounding_evaluation import calculate_metric_percase, statistic_results
    from evaluator.grounding_gen_evaluation import evaluate_json
    
    for batch in tqdm.tqdm(dataloader,
                           desc=f"Generating outputs (GPU {local_rank})" if is_distributed else "Generating outputs",
                           disable=not is_main_process()):
        
        output_texts, prediction = generate_text_and_grounding_from_batch(model, batch, max_new_tokens=1024, device="cuda")
        
        dataset_name = batch['dataset_name']
        sample_id = batch['sample_id']  # '16567_4_1_64'
        images = batch['seg_img']  # b d h w
        gt_segmentation = batch['seg_gt'].numpy() # b 1 h w   
        gt_answer = batch['answer_gt'] # list of batch_size string
        bs_label_ls = batch['label_ls']    # ls (batch size) of ls (num of abn in a sample/image), would be '' if no
        bs_type_ls_ls = batch['type_ls_ls']    # [[attribute1, attribute2, ...], ...]
        bs_description_ls = batch['description_ls']    # ls (batch size) of ls (num of abn in a sample/image), would be '' if no
        
        if args.visualization:
            rgb_img = images.numpy()
            rgb_img = repeat(rgb_img, 'b d h w -> b d h w c', c=3)
            rgb_pred = repeat(prediction, 'b o h w -> b h w (o c)', c=3)
            rgb_pred[:, :, :, 1:] = 0   # paint with R
            rgb_gt = repeat(gt_segmentation, 'b o h w -> b h w (o c)', c=3)
            rgb_gt[:, :, :, 1:] = 0     # paint with R
            
        # Calculation metrics and visualization
        
        for i in range(len(sample_id)):
            scores = calculate_metric_percase(prediction[i, :, :, :], gt_segmentation[i, :, :, :], True, False) # {'dice':0.9, 'nsd':0.8}
            results_of_samples.append([dataset_name[i], sample_id[i], [scores], bs_label_ls[i], bs_description_ls[i], bs_type_ls_ls[i], output_texts[i], gt_answer[i]])

            # visualization  
            if args.visualization:
                
                text = ''
                for j, (lbl, des, type_ls) in enumerate(zip(bs_label_ls[i], bs_description_ls[i], bs_type_ls_ls[i])):
                    type_str = ' / '.join(type_ls)
                    text += f'{j}: {lbl}; {des}; {type_str}'
                    text += '\n'
                
                txt_height = max(len(text)//60, 1)
                fig = plt.figure(figsize=(15, txt_height+5))
                gs = GridSpec(2, 3, figure=fig, height_ratios=[5, txt_height])
                
                # 第一排的三个子图
                img_tmp = rgb_img[i, rgb_img.shape[1]//2, :, :, :]    # h w 3
                img_tmp = (img_tmp - np.min(img_tmp)) / (np.max(img_tmp) - np.min(img_tmp)) # 0~1
                overlap_pred = img_tmp * 0.3 + rgb_pred[i, :, :, :] * 0.7
                overlap_gt = img_tmp * 0.3 + rgb_gt[i, :, :, :] * 0.7
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[0, 2])
                ax1.imshow(img_tmp)
                ax1.axis('off')  # 隐藏坐标轴
                ax2.imshow(overlap_pred)
                ax2.axis('off')  # 隐藏坐标轴
                ax3.imshow(overlap_gt)
                ax3.axis('off')  # 隐藏坐标轴
                
                # 第二排合并成一个子图，写入文本
                ax_text = fig.add_subplot(gs[1, :])  # 合并第二排的三列
                ax_text.text(0, 0.5, text, fontsize=14, va='center', ha='left', wrap=True)
                ax_text.axis('off')  # 隐藏坐标轴
                
                plt.tight_layout()
                plt.savefig(f'{visual_dir}/({round(scores["dice"], 2)}){dataset_name[i]}_{sample_id[i]}.jpg')
                
                # Create prediction_mask directory if it doesn't exist
                prediction_mask_dir = os.path.join(visual_dir, 'prediction_mask')
                os.makedirs(prediction_mask_dir, exist_ok=True)

                # Convert binary prediction to 0/255 image and save as jpg
                pred_mask = prediction[i, 0] * 255  # Convert from 0/1 to 0/255
                plt.figure(figsize=(5, 5))
                plt.imshow(pred_mask, cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'{prediction_mask_dir}/({round(scores["dice"], 2)}){dataset_name[i]}_{sample_id[i]}.jpg')
                plt.close()
                
                # Create input_image directory if it doesn't exist
                input_image_dir = os.path.join(visual_dir, 'input_image')
                os.makedirs(input_image_dir, exist_ok=True)

                # Save the input image as jpg
                input_img = img_tmp * 255  # Scale to 0-255 range
                plt.figure(figsize=(5, 5))
                plt.imshow(input_img)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'{input_image_dir}/({round(scores["dice"], 2)}){dataset_name[i]}_{sample_id[i]}.jpg')
                plt.close()
                
        # save in each process regularly in case of interruption
        if len(results_of_samples) % args.save_interval == 0:
            with open(csv_path.replace('.csv', f'_tmp_rank{get_rank()}.pkl'), 'wb') as f:
                pickle.dump(results_of_samples, f)
        
    torch.cuda.empty_cache()    
        
    # save grounding results in each process (to a fnl pickle, also denoting this process ends, del tmp pickle if existing)
    with open(csv_path.replace('.csv', f'_fnl_rank{get_rank()}.pkl'), 'wb') as f:
        pickle.dump(results_of_samples, f)
    if os.path.exists(csv_path.replace('.csv', f'_tmp_rank{get_rank()}.pkl')):
        os.remove(csv_path.replace('.csv', f'_tmp_rank{get_rank()}.pkl'))
                
    # Gather and record in main process
    
    if is_main_process():
        
        # detect the finish of each process
        while True:
            all_process_finished = True
            for rank_id in range(torch.distributed.get_world_size()):
                if not os.path.exists(csv_path.replace('.csv', f'_fnl_rank{rank_id}.pkl')): # xxx_tmp_rankx.pkl
                    all_process_finished = False
                    break
            if all_process_finished:
                break
            else:
                time.sleep(10)
        
        # read results of each process (samples may be duplicated due to the even distribution of ddp, check)
        results_of_samples = []    
        for rank_id in range(torch.distributed.get_world_size()):
            fnl_results_file = csv_path.replace('.csv', f'_fnl_rank{rank_id}.pkl')
            with open(fnl_results_file, 'rb') as f:
                results_of_samples += pickle.load(f)
            
        # check duplication
        unique_set = set()
        deduplicated_results_of_samples = []
        for dataset_name, sample_id, scores, label_ls, description_ls, type_ls_ls, output_text, gt_answer in results_of_samples:
            if f'{dataset_name}/{sample_id}' not in unique_set:
                unique_set.add(f'{dataset_name}/{sample_id}')
                deduplicated_results_of_samples.append([dataset_name, sample_id, scores, label_ls, description_ls, type_ls_ls, output_text, gt_answer])
        results_of_samples = deduplicated_results_of_samples
        
        # Evaluate the generation results
        
        id2pred_gt = {sample_id:(output_text, gt_answer) for dataset_name, sample_id, scores, label_ls, description_ls, type_ls_ls, output_text, gt_answer in results_of_samples}

        with open(args.radiopedia_json, 'r') as f:
            source_data = json.load(f)
            
        for datum_idx, datum in source_data.items():
            if datum_idx in id2pred_gt:
                datum['grounding_generation_results'] = id2pred_gt[datum_idx][0]
                datum['grounding_generation_gt'] = id2pred_gt[datum_idx][1]
                
        rcd_json = f'{args.rcd_dir}/(detailed){args.rcd_file}.json'
        with open(rcd_json, 'w') as f:
            json.dump(source_data, f, indent=4, ensure_ascii=False)
        print(f'Save detailed generation output to {rcd_json}')
            
        evaluate_json(
            data_json=rcd_json, 
            detailed_output_json=rcd_json, 
            merged_output_json=f'{args.rcd_dir}/(merged){args.rcd_file}.json'
        )
        
        # Evaluate the grounding results
        
        results_of_samples = [[dataset_name, sample_id, scores, label_ls, description_ls, type_ls_ls] for dataset_name, sample_id, scores, label_ls, description_ls, type_ls_ls, output_text, gt_answer in results_of_samples]
        
        statistic_results(results_of_samples, csv_path)
            
                    
        
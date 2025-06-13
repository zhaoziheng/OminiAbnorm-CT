import torch
import argparse
import os

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
    
    parser = argparse.ArgumentParser(description='Inference script for OminiAbnorm-CT')
    # DATA
    parser.add_argument('--data_json', type=str, required=True, help='Path to the dataset JSON file')
    parser.add_argument('--out_json', type=str, required=True, help='JSON to save the prediction. Can be the same as data_json')
    # MODEL
    parser.add_argument('--model_name', type=str, required=True, help='Identify the prediction in out JSON')
    parser.add_argument('--model_path', type=str, help='Path to the base model')
    parser.add_argument('--adapter_path', type=str, help='Path to the LoRA adapter weights')
    parser.add_argument('--tokenizer_path', type=str, help='Path to a modified tokenizer (with <SEG> added as special token)')
    parser.add_argument('--other_trainable_param_path', type=str, help='To load seg model, lm_head, embed_tokens (all the orther trainable param beside lora layers).')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--do_eval', type=str2bool, default=True, help='Calculate metrics after inference')
    args = parser.parse_args()

    # Initialize distributed environment
    
    from utils.distribute import main_print, get_rank, get_world_size
    
    if 'RANK' in os.environ:
        is_distributed = True
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        is_main_process = (local_rank == 0)
        main_print(f"Initialized distributed process group. Rank: {get_rank()}, World Size: {get_world_size()}")
    else:
        is_distributed = False  # WARNING May not work properly without ddp
    
    # Model

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, AutoTokenizer

    model_path = args.model_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    # Load LoRA weights

    if args.adapter_path is not None:
        model.load_adapter(args.adapter_path)
    
    # Load processor and expand the vocabulary with <SEG> 
    # WARNING must give args.tokenizer_path if add <SEG> and modify the embedding 
    
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path, padding_side="left", use_fast=True)
    
    if args.tokenizer_path is not None: 
        processor.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            padding_side="left"
        )
        model.resize_token_embeddings(len(processor.tokenizer))
    
    # Load lm head, embedding
    
    if args.other_trainable_param_path is not None:
    
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
        
    # Dataset
    
    from dataset.test_dataset_vis_prmpt_gen import Vis_Prmt_Gen_Dataset, Vis_Prmt_Gen_Collator
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    test_dataset = Vis_Prmt_Gen_Dataset(args.data_json)
        
    collator = Vis_Prmt_Gen_Collator(processor)

    # Set up distributed sampler if using distributed
    if is_distributed:
        sampler = DistributedSampler(test_dataset, shuffle=False)
        sampler.set_epoch(0)
    else:
        sampler = None

    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
    
    def generate_text_from_batch(model, batch, max_new_tokens=1024, device="cuda"):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Generate text with the model
        with torch.no_grad():
            outputs = model.generate(**batch, 
                                    output_hidden_states=True,
                                    return_dict_in_generate=True,
                                    max_new_tokens=max_new_tokens)
            
            generated_ids = outputs.sequences
        
        # Trim the generated ids to remove the input ids
        trimmed_generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(batch["input_ids"], generated_ids)
        ]
        
        # Decode the output texts
        output_texts = processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return output_texts

    # Inference
    
    all_outputs = []
    all_ids = []
    
    for batch, batch_id in tqdm.tqdm(dataloader, 
                                    desc=f"Generating outputs (GPU {local_rank})" if is_distributed else "Generating outputs",
                                    disable=not is_main_process):
        outputs = generate_text_from_batch(model, batch, max_new_tokens=1024)
        all_outputs.extend(outputs)
        all_ids.extend(batch_id)
    
    # Gather results from all processes if in distributed mode
    if is_distributed:
        # Convert to tensors for gathering
        import pickle
        local_data = pickle.dumps((all_outputs, all_ids))
        local_size = torch.tensor(len(local_data), device="cuda")
        
        # Get sizes from all processes
        if is_main_process:
            sizes = [torch.tensor(0, device="cuda") for _ in range(get_world_size())]
        else:
            sizes = None
        
        # Gather sizes
        torch.distributed.gather(local_size, sizes if is_main_process else None, dst=0)
        
        # Broadcast max_size to all processes
        if is_main_process:
            max_size = max(sizes)
        else:
            max_size = torch.tensor(0, device="cuda")
        torch.distributed.broadcast(max_size, src=0)
            
        # Prepare buffers for data
        if is_main_process:
            gathered_data = [torch.zeros(max_size, dtype=torch.uint8, device="cuda") for _ in range(get_world_size())]
        else:
            gathered_data = None
            
        # Pad local data to max size
        padded_data = torch.zeros(max_size, dtype=torch.uint8, device="cuda")
        padded_data[:len(local_data)] = torch.tensor(list(local_data), dtype=torch.uint8, device="cuda")
        
        # Gather all data
        torch.distributed.gather(padded_data, gathered_data if is_main_process else None, dst=0)
        
        # Combine results on the main process
        if is_main_process:
            all_outputs = []
            all_ids = []
            for i, (size, data) in enumerate(zip(sizes, gathered_data)):
                outputs, ids = pickle.loads(data[:size].cpu().numpy().tobytes())
                all_outputs.extend(outputs)
                all_ids.extend(ids)
    
    # Evaluation - only on the main process
    if args.do_eval and (is_main_process or not is_distributed):
        
        # Save to json
        
        import json
        import os
        from pathlib import Path
    
        id2pred = {label.strip(): pred.strip() for pred, label in zip(all_outputs, all_ids)}
            
        if os.path.exists(args.out_json):
            with open(args.out_json, "r", encoding="utf-8") as f:
                source_data = json.load(f)
        else:
            with open(args.data_json, "r", encoding="utf-8") as f:
                source_data = json.load(f)
            
        for sample_id, content in source_data.items():
            for prompt_type in ['bbox', 'ellipse']:
                if f'{sample_id}/negative/{prompt_type}' in id2pred:
                    content[f'{args.model_name}_{prompt_type}_neg_prmpt_answer'] = id2pred[f'{sample_id}/negative/{prompt_type}'] 
            
            for abnormality_content in content['abnormality']:
                abnormality_id = abnormality_content['id']
                for prompt_type in ['bbox', 'contour', 'cropped', 'ellipse']:
                    if f'{sample_id}/{abnormality_id}/{prompt_type}' in id2pred:
                        abnormality_content[f'{args.model_name}_{prompt_type}_answer'] = id2pred[f'{sample_id}/{abnormality_id}/{prompt_type}']
               
            # if 'abnormality_group' in content:
            #     for group_id, group in content['abnormality_group'].items():
            #         id_ls = group['id']
            #         abnormality_id = '_'.join(str(id) for id in id_ls)
            #         for prompt_type in ['bbox', 'contour', 'cropped', 'ellipse']:
            #             if f'{sample_id}/{abnormality_id}/{prompt_type}' in id2pred:
            #                 group[f'{args.model_name}_{prompt_type}_answer'] = id2pred[f'{sample_id}/{abnormality_id}/{prompt_type}']
               
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)        
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(source_data, f, ensure_ascii=False, indent=4)
            
        # Evaluate from json
            
        from evaluator.vis_prmt_gen_evaluation import evaluate_json
        
        merged_output_json = os.path.join(
            os.path.dirname(args.out_json),
            f"(merged){os.path.basename(args.out_json).replace('(detailed)', '')}"
        )

        evaluate_json(
            args.out_json, 
            args.model_name, 
            args.out_json, 
            merged_output_json
        )
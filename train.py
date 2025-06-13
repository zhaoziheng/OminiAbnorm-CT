import os
import json
from datetime import datetime

proxy = 'http://zhaoziheng:aJTjJb3qJJIhkd9uui0tnFQLUCsChFwTRsDluzv5ldgo8dQWu834Ac4rQbba@10.1.20.50:23128/'
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

if __name__ == '__main__':
    
    # Do this if you encounter annoying warnings from torchvision transform
    
    import warnings
    warnings.filterwarnings("ignore", message="The default value of the antialias parameter")
    warnings.filterwarnings("ignore", message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta")
    
    # Initialize distributed environment for DDP training
    
    import torch
    from utils.distribute import main_print, get_rank, get_world_size, is_main_process
    
    if 'RANK' in os.environ:
        is_distributed = True
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        main_print(f"Initialized distributed process group. Rank: {get_rank()}, World Size: {get_world_size()}")
    else:
        is_distributed = False  # WARNING May not work properly without ddp
        
    # Parse configuration from YAML file or command line (allow override)
    
    from utils.config import get_train_config
    
    args = get_train_config()
    
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime('%Y-%m-%d_%H-%M')
    args.exp_name = os.path.join('./log', args.exp_name)
    main_print(f"** TRAINER ** Training output to {args.exp_name}")
    
    os.makedirs(args.exp_name, exist_ok=True)

    args_dict = vars(args)
    with open(os.path.join(args.exp_name, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)
        
    # Backup the code
    
    from utils.code_backup import backup_code
    
    if is_main_process():
        backup_code(
            code_base = './',
            target_file_suffix = ['.py'],
            keyword_filter = ['log', 'deprecated', 'wandb'],
            backup_dir = os.path.join(args.exp_name, 'code')
        )

    # Load Model
    
    import torch
    from transformers import Qwen2_5_VLProcessor, AutoTokenizer
    from model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_wSEG

    model = Qwen2_5_VLForConditionalGeneration_wSEG.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    # Load processor or add <SEG>
    
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path, padding_side="left", use_fast=True)
    if args.tokenizer_path is not None:
        processor.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.tokenizer_path),
            padding_side="left"
        )
    else:
        special_tokens_dict = {'additional_special_tokens': ['<SEG>']}
        processor.tokenizer.add_special_tokens(special_tokens_dict)
        processor.tokenizer.save_pretrained(os.path.join(args.exp_name, "tokenizer"))
    
    # Configure LoRA

    from peft import LoraConfig

    if args.adapter_path is not None:
        main_print(f"** ADAPTER ** Loading adapter from {args.adapter_path}")
        model.load_adapter(args.adapter_path)
        # Freeze base model weights and only train the adapter
        for name, param in model.named_parameters():
            if 'lora' not in name.lower():  # Freeze parameters that are not part of LoRA
                param.requires_grad = False
            else:
                param.requires_grad = True  # Make sure LoRA parameters are trainable
    else:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=args.lora_rank,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model.add_adapter(peft_config)
        
    # Expand the vocabulary with <SEG>
    
    model.resize_token_embeddings(len(processor.tokenizer)) # make sure resize after adding to processor
    model.seg_token_idx = processor.tokenizer.convert_tokens_to_ids('<SEG>')
    
    # Some params in VLM should also be optimized (beside lora layers)
    
    if args.other_trainable_param_path is not None:
        main_print(f"** MODEL ** Try to loading some weights from {args.other_trainable_param_path}")
        other_trainable_param_ckpt = torch.load(args.other_trainable_param_path)
    else:
        other_trainable_param_ckpt = []
    
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
        
    # Add Segmentation Module (Must after PEFT Config or the Seg Model wont be trainable)
    
    from model.segmentation_loss import SegmentationLoss
    from model.unet import UNET
    
    model.seg_loss = SegmentationLoss()
    model.add_module('seg_model', UNET(input_channels=9))
    model.seg_model = model.seg_model.to(get_rank())
    
    if args.other_trainable_param_path is not None:
        main_print(f"** SEG MODEL ** Loaded from checkpoint: {args.other_trainable_param_path}")
        model.seg_model.load_saved(other_trainable_param_ckpt)
        
    model.seg_model.frozen(args.forzen_seg_model)
        
    # Monitor Seg Loss without Modifying Trainer
        
    from utils.meter import AverageMeter    
    
    model.dice_loss_m = AverageMeter()
    model.ce_loss_m = AverageMeter()
    
    # Show trainable parameters
        
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        if 'seg_model' in name:
            continue
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    main_print(f"** VLM MODEL ** Trainable parameters: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}% of {all_params:,d} total params)")

    trainable_params = 0
    all_params = 0
    for name, param in model.seg_model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    main_print(f"** SEG MODEL ** Trainable parameters: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}% of {all_params:,d} total params)")
    
    # Load Dataset
    
    from dataset.train_dataset_vis_prmpt_gen import Vis_Prmt_Gen_Dataset
    from dataset.train_dataset_grounding import Grounding_Dataset
    from dataset.train_dataset_refer_grounding import Refer_Grounding_Two_Round_Dataset, Refer_Grounding_One_Round_Dataset 
    from dataset.train_dataset_grounding_gen import Grounding_Gen_Two_Round_Dataset, Grounding_Gen_One_Round_Dataset
    from dataset.train_dataset_instruction import Instruction_QA_Dataset
    from dataset.multi_dataset import MultiTypeDataset
    
    dataset1 = Vis_Prmt_Gen_Dataset(
        json_path=args.train_data_json, 
        rewritten_max=args.rewritten_max,  # take GPT rewritten results as gt
        allow_multi_prompt=args.allow_multi_prompt,  # whether to use multi prompt for abnormality group
        epoch_quota=args.vis_prmpt_gen_epoch_quota, # Quota for this dataset in an epoch
    )
    
    dataset2 = Grounding_Dataset(
        deeplesion_json=args.deeplesion_json,
        radiopedia_json=args.radiopedia_json,
        other_data_json=args.other_data_json,
        deeplesion_weight=args.deeplesion_weight,
        radiopedia_weight=args.radiopedia_weight,
        other_data_weight=args.other_data_weight,
        epoch_quota=args.grounding_epoch_quota, # Quota for this dataset in an epoch
    )
    
    dataset3 = Refer_Grounding_Two_Round_Dataset(
        deeplesion_json=None,
        radiopedia_json=args.radiopedia_json,
        other_data_json=None,
        deeplesion_weight=0,
        radiopedia_weight=1,
        other_data_weight=0,
        negative_prompt_ratio=args.negative_refer_ratio,
        epoch_quota=args.refer_grounding_epoch_quota, # Quota for this dataset in an epoch
    )
    
    dataset4 = Refer_Grounding_One_Round_Dataset(
        deeplesion_json=None,
        radiopedia_json=args.radiopedia_json,
        other_data_json=None,
        deeplesion_weight=0,
        radiopedia_weight=1,
        other_data_weight=0,
        negative_prompt_ratio=args.negative_refer_ratio,
        epoch_quota=args.refer_grounding_epoch_quota, # Quota for this dataset in an epoch
    )
    
    dataset5 = Grounding_Gen_Two_Round_Dataset(
        deeplesion_json=None,
        radiopedia_json=args.radiopedia_json,
        other_data_json=None,
        deeplesion_weight=0,
        radiopedia_weight=1,
        other_data_weight=0,
        epoch_quota=args.grounding_gen_epoch_quota, # Quota for this dataset in an epoch
    )
    
    dataset6 = Grounding_Gen_One_Round_Dataset(
        deeplesion_json=None,
        radiopedia_json=args.radiopedia_json,
        other_data_json=None,
        deeplesion_weight=0,
        radiopedia_weight=1,
        other_data_weight=0,
        epoch_quota=args.grounding_gen_epoch_quota, # Quota for this dataset in an epoch
    )
    
    dataset7 = Instruction_QA_Dataset(
        pubmedvision_json=args.pubmedvision_json,
        epoch_quota=args.instruction_qa_epoch_quota,
    )
    
    # train_dataset = MultiTypeDataset(
    #     datasets=[dataset1, dataset2, dataset3, dataset4],
    #     batch_size=args.per_device_batch_size,
    #     seed=42
    # )
    train_dataset = MultiTypeDataset.from_distributed_env([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7], args.per_device_batch_size, seed=42)
    
    from dataset.multi_collator import MultiTypeCollator
    
    collator = MultiTypeCollator(processor)
    
    # SFT Training Config

    from trl import SFTConfig

    training_args = SFTConfig(
        output_dir=args.exp_name,  # Directory to save the model
        num_train_epochs=args.epochs,  # Number of training epochs from args
        per_device_train_batch_size=args.per_device_batch_size,  # Batch size for training from args
        per_device_eval_batch_size=args.per_device_batch_size,  # Batch size for evaluation from args
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Steps to accumulate gradients from args
        # Gradient checkpointing settings
        gradient_checkpointing=args.gradient_checkpointing,  # Enable gradient checkpointing for memory efficiency
        ddp_find_unused_parameters=False, # This is necessary for gradient_checkpointing=True
        # gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=args.logging_steps,  # Steps interval for logging from args
        eval_steps=args.eval_steps,  # Steps interval for evaluation from args
        eval_strategy="steps" if args.valid_data_json is not None else 'no',  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=args.save_steps,  # Steps interval for saving from args
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True if args.valid_data_json is not None else False,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.05,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        report_to=None, # "wandb",  # Reporting tool for tracking metrics
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options    # NOTE 不使用HF默认的数据处理
        max_seq_length=1024,
        dataset_num_proc=32,
        dataloader_num_workers=16,
        # dataset_num_proc=1, 
        # dataloader_num_workers=1,   
        dataloader_pin_memory=True,
        dataloader_shuffle=False,  # disable shuffle so MultiTypeDataset could arrange data from the same task in a batch
        )
    
    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    # Trainer

    from utils.sft_trainer import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collator,
        # peft_config=peft_config,  # NOTE This is not needed if load adapter with peft
        # tokenizer=processor.tokenizer,
        # dataset_num_proc=32,
    )
    
    # Add 3 callbacks:
    # 1. Shuffle batches before every epoches
    # 2. Save trainable paramters (except for lora layers) regularly  (On main process)
    # 3. Log segmentation loss regularly  (On main process)
    
    from transformers import TrainerCallback
    
    class EpochCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            if hasattr(trainer.train_dataset, 'set_epoch'):
                trainer.train_dataset.set_epoch(state.epoch)
                
    trainer.add_callback(EpochCallback())
                    
    class CustomSaveCallback(TrainerCallback):
        def __init__(self, save_steps, out_dir):
            self.save_steps = save_steps
            self.out_dir = out_dir
            
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if args.local_rank == 0 or args.local_rank == -1:
                if state.global_step % self.save_steps == 0:
                    # seg_model
                    combined_state_dict = {
                        'seg_model':trainer.model.seg_model.state_dict(),
                        }
                    # learnable parameters in LLM
                    lm_head_state_dict = {}
                    embed_tokens_state_dict = {}
                    for name, param in trainer.model.named_parameters():
                        if "lm_head" in name:
                            lm_head_state_dict[name] = param
                        elif "embed_tokens" in name:
                            embed_tokens_state_dict[name] = param
                    combined_state_dict['lm_head_state_dict'] = lm_head_state_dict
                    combined_state_dict['embed_tokens_state_dict'] = embed_tokens_state_dict
                    # save
                    save_path = os.path.join(self.out_dir, f'checkpoint-{state.global_step}')
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(combined_state_dict, os.path.join(save_path, 'seg_model_lm_head_embed_tokens.pt'))
                    
        def on_train_end(self, args, state, control, model=None, **kwargs):
            # Ensure saving on the final step regardless of save_steps interval
            if args.local_rank == 0 or args.local_rank == -1:
                # seg_model
                combined_state_dict = {
                    'seg_model':model.seg_model.state_dict(),
                    }
                # learnable parameters in LLM
                lm_head_state_dict = {}
                embed_tokens_state_dict = {}
                for name, param in model.named_parameters():
                    if "lm_head" in name:
                        lm_head_state_dict[name] = param
                    elif "embed_tokens" in name:
                        embed_tokens_state_dict[name] = param
                combined_state_dict['lm_head_state_dict'] = lm_head_state_dict
                combined_state_dict['embed_tokens_state_dict'] = embed_tokens_state_dict
                # save
                save_path = os.path.join(self.out_dir, f'checkpoint-final')
                os.makedirs(save_path, exist_ok=True)
                torch.save(combined_state_dict, os.path.join(save_path, 'seg_model_lm_head_embed_tokens.pt'))
                    
    trainer.add_callback(CustomSaveCallback(args.save_steps, args.exp_name))
    
    class CustomLogCallback(TrainerCallback):
        def __init__(self, logging_steps, out_dir):
            self.logging_steps = logging_steps
            self.out_dir = out_dir
            
        def on_step_end(self, args, state, control, **kwargs):
            if args.local_rank == 0 or args.local_rank == -1:
                if state.global_step % self.logging_steps == 0:
                    save_path = os.path.join(self.out_dir, 'segmentation_loss_log.jsonl')
                    with open(save_path, 'a') as f:
                        log_data = {
                            'step': state.global_step,
                            'dice_loss': model.dice_loss_m.avg,
                            'ce_loss': model.ce_loss_m.avg
                        }
                        f.write(json.dumps(log_data) + '\n')
                    model.ce_loss_m.reset()
                    model.dice_loss_m.reset()
                
    trainer.add_callback(CustomLogCallback(args.logging_steps, args.exp_name))

    # Start Training

    trainer.train()
import yaml
import argparse

def get_train_config():
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def load_yaml_config(yaml_file):
        """Load YAML configuration file"""
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

    parser = argparse.ArgumentParser(description='Training script for OminiAbnorm-CT')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    config_args, remaining_argv = parser.parse_known_args()
    
    config = {}
    if config_args.config:
        config = load_yaml_config(config_args.config)
    
    # Task
    parser.add_argument('--exp_name', type=str, default=config.get('exp_name', None), help='Dir to save the LORA weights')
    # Vis Prmpt Gen Task
    parser.add_argument('--train_data_json', type=str, default=config.get('train_data_json'), help='Path to the train set JSON file')
    parser.add_argument('--rewritten_max', type=int, default=config.get('rewritten_max', 3), help='Per Sample. Take GPT rewritten results as gt')
    parser.add_argument('--allow_multi_prompt', type=str2bool, default=config.get('allow_multi_prompt', True), help='Whether to use multi visual prompt for abnormality group (same abnormality but scattered)')
    parser.add_argument('--vis_prmpt_gen_epoch_quota', type=int, default=config.get('vis_prmpt_gen_epoch_quota'), help='Determine the number of samples in one epoch')
    parser.add_argument('--negative_prmpt_ratio', type=float, default=config.get('negative_prmpt_ratio'), help='Prob to sample negative prompts')
    # Grounding Task
    parser.add_argument('--deeplesion_json', type=str, default=config.get('deeplesion_json'), help='Path to DeepLesion dataset JSON file')
    parser.add_argument('--radiopedia_json', type=str, default=config.get('radiopedia_json'), help='Path to Radiopedia dataset JSON file')
    parser.add_argument('--other_data_json', type=str, default=config.get('other_data_json'), help='Path to Public Lesion Segmentation datasets JSON file')
    parser.add_argument('--deeplesion_weight', type=float, default=config.get('deeplesion_weight'), help='Weight to sample DeepLesion data')
    parser.add_argument('--radiopedia_weight', type=float, default=config.get('radiopedia_weight'), help='Weight to sample Radiopedia data')
    parser.add_argument('--other_data_weight', type=float, default=config.get('other_data_weight'), help='Weight to sample other data')
    parser.add_argument('--grounding_epoch_quota', type=int, default=config.get('grounding_epoch_quota'), help='Determine the number of samples in one epoch')
    # Refer Grounding Task
    parser.add_argument('--negative_refer_ratio', type=float, default=config.get('negative_refer_ratio'), help='Prob to sample negative refer')
    parser.add_argument('--refer_grounding_epoch_quota', type=int, default=config.get('refer_grounding_epoch_quota'), help='Determine the number of samples in one epoch')
    # Grounding Gen Task
    parser.add_argument('--grounding_gen_epoch_quota', type=int, default=config.get('grounding_gen_epoch_quota'), help='Determine the number of samples in one epoch')
    # Instruction QA Task
    parser.add_argument('--pubmedvision_json', type=str, default=config.get('pubmedvision_json'), help='Path to PubMedVision dataset JSON file')
    parser.add_argument('--instruction_qa_epoch_quota', type=int, default=config.get('instruction_qa_epoch_quota'), help='Determine the number of samples in one epoch')
    # Model
    parser.add_argument('--model_path', type=str, default=config.get('model_path', "x"), help='Path to the model')
    parser.add_argument('--adapter_path', type=str, default=config.get('adapter_path'), help='Path to the adapter weights, used to resume training')
    parser.add_argument('--lora_rank', type=int, default=config.get('lora_rank', 32), help='Rank for LoRA adaptation')
    parser.add_argument('--tokenizer_path', type=str, default=config.get('tokenizer_path'), help='Path to a modified tokenizer (with <SEG> added as special token)')
    parser.add_argument('--other_trainable_param_path', type=str, default=config.get('other_trainable_param_path'), help="""Used to resume training.
                                                                                                                            Can either be ckpt from segmentation pre-training (only try to load the seg model part); 
                                                                                                                            Or ckpt with seg model, lm_head, embed_tokens (all the orther trainable param beside lora layers).""")
    parser.add_argument('--forzen_seg_model', type=str, nargs='+', default=config.get('forzen_seg_model'), help='Freeze some part of the Segmentation model. Refer to frozen() in UNET.')
    # Training
    parser.add_argument('--epochs', type=int, default=config.get('epochs', 3), help='Number of training epochs')
    parser.add_argument('--logging_steps', type=int, default=config.get('logging_steps', 2000), help='Steps interval for logging')
    parser.add_argument('--eval_steps', type=int, default=config.get('eval_steps', 2000), help='Steps interval for evaluation')
    parser.add_argument('--save_steps', type=int, default=config.get('save_steps', 4000), help='Steps interval for saving checkpoints')
    parser.add_argument('--per_device_batch_size', type=int, default=config.get('per_device_batch_size', 2), help='Batch size per GPU device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=config.get('gradient_accumulation_steps', 4), help='Number of steps to accumulate gradients before updating weights')
    parser.add_argument('--gradient_checkpointing', type=str2bool, default=config.get('gradient_checkpointing', True), help='Whether to use gradient_checkpointing')
    parser.add_argument('--valid_data_json', type=str, default=config.get('valid_data_json'), help='Path to the validation set JSON file')
    
    # Parse the remaining arguments
    args = parser.parse_args(remaining_argv)
    
    return args
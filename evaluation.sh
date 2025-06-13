cd OminiAbnorm-CT

torchrun --nproc_per_node=8 --master_port 25724 train.py \
--config 'train.yaml'

####################################
# 1️ Visual Prompted Generation Task
####################################

torchrun --nproc_per_node=1 --master_port 25813 evaluate_vis_prmpt_gen.py \
--data_json 'axial/axial_valid.json' \
--out_json './checkpoints/ominiabnorm-ct-7b/results_axial/(detailed)vis_prmpt_gen(axial).json' \
--model_name 'OminiAbnorm-CT' \
--model_path 'Qwen/Qwen2.5-VL-7B-Instruct' \
--tokenizer_path './checkpoints/ominiabnorm-ct-7b-latest/extended_tokenizer' \
--adapter_path './checkpoints/ominiabnorm-ct-7b-latest' \
--other_trainable_param_path './checkpoints/ominiabnorm-ct-7b-latest/seg_model_lm_head_embed_tokens.pt' \

####################################################
# 2 Refer Grounding Task (Should use _grouped.json)
####################################################

torchrun --nproc_per_node=4 --master_port 25129 evaluate_refer_grounding.py \
--radiopedia_json 'axial/axial_valid_grouped.json' \
--rcd_dir './checkpoints/ominiabnorm-ct-7b/results_axial/' \
--model_path 'Qwen/Qwen2.5-VL-7B-Instruct' \
--tokenizer_path './checkpoints/ominiabnorm-ct-7b-latest/extended_tokenizer' \
--adapter_path './checkpoints/ominiabnorm-ct-7b-latest' \
--other_trainable_param_path './checkpoints/ominiabnorm-ct-7b-latest/seg_model_lm_head_embed_tokens.pt' \
--visualization False

##############################
# 3 Grounding Generation Task
##############################

torchrun --nproc_per_node=4 --master_port 25233 evaluate_grounding_gen.py \
--radiopedia_json 'axial/axial_valid.json' \
--rcd_dir './checkpoints/ominiabnorm-ct-7b/results_axial/' \
--model_path 'Qwen/Qwen2.5-VL-7B-Instruct' \
--tokenizer_path './checkpoints/ominiabnorm-ct-7b-latest/extended_tokenizer' \
--adapter_path './checkpoints/ominiabnorm-ct-7b-latest' \
--other_trainable_param_path './checkpoints/ominiabnorm-ct-7b-latest/seg_model_lm_head_embed_tokens.pt' \
--visualization False
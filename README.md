# OminiAbnorm-CT

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv)](https://www.arxiv.org/abs/2506.03238)
[![HF](https://img.shields.io/badge/Hugging%20Face-Data-yellow)](https://huggingface.co/datasets/zzh99/OminiAbnorm-CT-14K)
[![HF](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/zzh99/OminiAbnorm-CT/blob/main/README.md)

This is the official repository for "Rethinking Whole-Body CT Image Interpretation: An Abnormality-Centric Approach".

**OminiAbnorm-CT** is developed for grounded abnormality analysis on CT images from multiple planes and all human body regions.
It support three representative task:
- Visual prompted report generation: Interpretate an abnormality marked by red bounding box, ellipse, contour, or cropped region.
- Grounded report generation: Ground and interpretate all abnormalities on the CT image.
- Text-guided grounded report generation: Detect, ground and interpretate specific abnormality on the CT image.

![Example Figure](docs/resources/model.png)

It is built on **OminiAbnorm-CT-14K**, the first large-scale dataset designed for abnormality grounding and description on multi-plane whole-body CT imaging. 
It contains 14.5K CT images with grounding annotation for 19K abnormal findings.
Each abnormal findings is further linked to the detailed description in the report, and categorized according to a comprehensive hierarchical taxonomy.

![Example Figure](docs/resources/data.png)

🔍 Check our [paper](https://www.arxiv.org/abs/2506.03238) for more details.

## Data
(Essential) Check [here](https://huggingface.co/datasets/zzh99/OminiAbnorm-CT-14K) for the OminiAbnorm-CT-14K dataset and the taxonomy.

(Optional) Other datasets that can be utilized in training:
- Dataset for lesion detection/segmentation on CT images: [DeepLesion](https://paperswithcode.com/dataset/deeplesion), [ULS23](https://uls23.grand-challenge.org), [MSD](http://medicaldecathlon.com), [KiTS23](https://kits-challenge.org/kits23/), etc. For a detailed list, refer to Table 6 in our paper. We processed them into 2D segmentation data, maintaining the same format as OminiAbnorm-CT-14K. For detailed processing procedures, please refer to `dataset/process_other_datasets.py`. Note that they have no reports, and some even lack category labels. 
- Dataset for general medical VQA: [PubmedVison](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision). Specifically, the CT images in it.

## Checkpoint
- Download the model checkpoint from [huggingface](https://huggingface.co/zzh99/OminiAbnorm-CT/blob/main/README.md) to `./checkpoint/`.
- (Optinal) Download the base model [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

## Requirements
Create a new environment:
```
conda create -n your_env_name python=3.10
conda activate your_env_name
```
Install all the requirements (some packages may not be necessary):
```
pip install -r requirements.txt
```
Some packages that need particular attention:
```
torch==2.0.0
transformers==4.49.0
trl==0.16.1
flash-attn==2.3.6
monai==1.3.0
scipy==1.12.0
# To evaluate the generated report
evaluate==0.4.3
RaTEScore==0.5.0
radgraph==0.1.17
```
In addition, to support the segmentation model, you need to install a customized version of [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures/tree/main):
```
cd model
pip install -e dynamic-network-architectures-main
```

## Inference Guidance
Place your CT images in a folder (named in the format 0.jpg, 1.jpg, ··· to indicate their sequence). By default, we will select the middle slice and 8 adjacent slices (if segmentation model is invoked) for analysis. Alternatively, you can directly provide the path to a single CT image slice.

Run:
```
python inference.py
```
We also provide 3 demo cases (one for each task) in the script for your reference. 

Note that to mark an abnormality with visual prompt, directly draw on the image (box, circle, ellipse, outline, contour) or save the cropped region as a image file.


## Traning Guidance
Prepare your training configuration in `train.yaml` and continue fine-tuning on our checkpoint:
```
torchrun --nproc_per_node=8 --master_port xxxxx train.py --config 'train.yaml'
```
Or, train from scratch based on Qwen2.5-VL-7B and the U-Net pre-trained on CT lesion segmentation:
```
torchrun --nproc_per_node=8 --master_port xxxxx train.py --config 'train_from_scratch.yaml'
```

## Evaluation Guidance
We provide detailed evaluation script for each task in `evaluation.sh`.

## Citation
```
@misc{zhao2025rethinkingwholebodyctimage,
      title={Rethinking Whole-Body CT Image Interpretation: An Abnormality-Centric Approach}, 
      author={Ziheng Zhao and Lisong Dai and Ya Zhang and Yanfeng Wang and Weidi Xie},
      year={2025},
      eprint={2506.03238},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2506.03238}, 
}
```

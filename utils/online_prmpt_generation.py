import torch
import numpy as np
import cv2
from typing import List, Tuple
from qwen_vl_utils import process_vision_info
from PIL import Image

def formulate_second_round_prompt(first_round_content, output_text, vis_prmpt_img):
    output = first_round_content + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": output_text,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "segmentation_tool",
                "content": [
                    {
                        "type": "image",
                        "image": vis_prmpt_img,
                    }
                ],
            }
            # {
            #     "role": "user",
            #     "content": [
            #         {
            #             "type": "image",
            #             "image": vis_prmpt_img,
            #         },
            #         {
            #         "type": "text",
            #         "text": 'This is the detection result. Please check it.',
            #         }
            #     ],
            # }
        ]
    return output
    
def generate_second_round_input(processor, first_round_vllm_raw_inputs, output_texts, vis_prmpt_img_ls):
        chat_format_data = [formulate_second_round_prompt(f, t, v) for f, t, v in zip(first_round_vllm_raw_inputs, output_texts, vis_prmpt_img_ls)]

        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(datum, tokenize=False, add_generation_prompt=True) for datum in chat_format_data]  # Prepare texts for processing
        image_inputs = [process_vision_info(datum)[0] for datum in chat_format_data]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True, truncation=True)  # Encode texts and images into tensors
        
        return batch  # Return the prepared batch
    
def draw_bounding_box_w_random_shift(image, sc_mask, color=(0, 0, 255), thickness=2, shift_range=0.25):
    """
    Draw a bounding box around the non-zero region in sc_mask on the input image.

    Args:
        image (np.ndarray): Input image of shape (H, W).
        sc_mask (np.ndarray): Binary mask of shape (H, W) indicating the region of interest.
        color (tuple): Color of the bounding box in BGR format (default: red).
        thickness (int): Thickness of the bounding box lines (default: 2).
        shift_range (float): Range for random shift of the bounding box (default: 0.1 of the edge length).

    Returns:
        np.ndarray: Image with the bounding box drawn.
    """
    
    image = np.array(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
    sc_mask = np.array(sc_mask)
    boxes = segmentation_post_process(sc_mask)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Generate random shifts for each coordinate
        x1_shift = int(np.random.uniform(-shift_range, shift_range) * width)
        y1_shift = int(np.random.uniform(-shift_range, shift_range) * height)
        x2_shift = int(np.random.uniform(-shift_range, shift_range) * width)
        y2_shift = int(np.random.uniform(-shift_range, shift_range) * height)

        # Apply shifts
        x1 = max(0, x1 + x1_shift)
        y1 = max(0, y1 + y1_shift)
        x2 = min(image_rgb.shape[1], x2 + x2_shift)
        y2 = min(image_rgb.shape[0], y2 + y2_shift)

        # Ensure box dimensions are valid
        if x2 <= x1:
            x1, x2 = x2 - 1, x1 + 1
        if y2 <= y1:
            y1, y2 = y2 - 1, y1 + 1
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, thickness)
        
    return image_rgb

def draw_bounding_box(image, sc_mask, color=(0, 0, 255), thickness=2):
    """
    Draw a bounding box around the non-zero region in sc_mask on the input image.

    Args:
        image (np.ndarray): Input image of shape (H, W).
        sc_mask (np.ndarray): Binary mask of shape (H, W) indicating the region of interest.
        color (tuple): Color of the bounding box in BGR format (default: red).
        thickness (int): Thickness of the bounding box lines (default: 2).

    Returns:
        PIL.Image: Image with the bounding box drawn.
    """
    
    image = np.array(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
    sc_mask = np.array(sc_mask)
    boxes = segmentation_post_process(sc_mask)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, thickness)
        
    return image_rgb

def segmentation_post_process(
    mask,
    min_area: int = 16,
    iou_threshold: float = 0.5,
    morph_kernel_size: int = 5
) -> List[List[Tuple[int, int, int, int]]]:
    """
    将二值分割掩码转换为检测框，包含后处理步骤
    
    Args:
        mask: 输入二值分割掩码，形状为 (h, w)
        min_area: 最小区域面积阈值，小于此值的区域会被过滤
        iou_threshold: NMS合并时的IoU阈值
        morph_kernel_size: 形态学操作的核大小
    
    Returns:
        List[List[Tuple[x1, y1, x2, y2]]]: 每个batch的检测框列表
    """
    # 确保输入是PyTorch Tensor或numpy数组
    if isinstance(mask, torch.Tensor):
        mask_np = mask.squeeze(1).detach().cpu().numpy()  # (b, h, w)
        
    mask_np = mask
    
    # 二值化处理（假设输入已经是0/1，这里做保险）
    binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
    
    # 形态学后处理（可选）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        processed_mask, connectivity=8, ltype=cv2.CV_32S
    )
    
    # 提取候选框 (跳过背景标签0)
    boxes = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        
        # 过滤小区域
        if area < min_area:
            continue
            
        boxes.append([x, y, x + w, y + h])  # (x1, y1, x2, y2)格式
    
    # 非极大值抑制合并重叠框
    if len(boxes) > 0:
        boxes = np.array(boxes)
        keep_indices = nms(boxes, iou_threshold)
        boxes = boxes[keep_indices].tolist()
    
    return [tuple(box) for box in boxes]

def nms(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    NMS实现（按面积从大到小排序后合并）
    
    Args:
        boxes: (n, 4)格式的numpy数组，每行是[x1, y1, x2, y2]
        iou_threshold: 合并阈值
    
    Returns:
        保留的框索引
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    
    # 计算每个框的面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 按面积降序排序
    sorted_indices = np.argsort(-areas)
    keep = []
    
    while len(sorted_indices) > 0:
        # 取当前最大的框
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        
        # 计算与其他框的IoU
        current_box = boxes[current_idx]
        other_boxes = boxes[sorted_indices[1:]]
        
        xx1 = np.maximum(current_box[0], other_boxes[:, 0])
        yy1 = np.maximum(current_box[1], other_boxes[:, 1])
        xx2 = np.minimum(current_box[2], other_boxes[:, 2])
        yy2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[current_idx] + areas[sorted_indices[1:]] - inter
        iou = inter / union
        
        # 保留IoU低于阈值的框
        remaining_indices = np.where(iou <= iou_threshold)[0]
        sorted_indices = sorted_indices[remaining_indices + 1]  # +1因为前面取了[1:]
    
    return np.array(keep)

# 测试示例
if __name__ == "__main__":
    # 模拟输入 (batch_size=2, 1 channel, height=100, width=100)
    dummy_mask = torch.zeros((2, 1, 100, 100))
    
    # Batch 0: 两个不重叠的矩形
    dummy_mask[0, 0, 10:30, 20:40] = 1
    dummy_mask[0, 0, 50:80, 60:90] = 1
    
    # Batch 1: 两个重叠的矩形 + 噪声
    dummy_mask[1, 0, 30:50, 10:30] = 1
    dummy_mask[1, 0, 40:60, 20:40] = 1
    dummy_mask[1, 0, 80:85, 80:85] = 1  # 小区域（会被过滤）
    
    boxes = segmentation_post_process(dummy_mask)
    print("Output boxes:")
    for i, batch_boxes in enumerate(boxes):
        print(f"Batch {i}: {batch_boxes}")
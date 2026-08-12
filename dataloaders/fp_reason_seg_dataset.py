import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json_v2

from .qa_template import LONG_QUESTION_TEMPLATE, LONG_ANSWER_TEMPLATE, SHORT_QUESTION_TEMPLATE

class fpReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    choices = ["True_Premise",  "False_Premise_Correction"]  
    weights = [0.85, 0.15]
    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 336,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        num_classes_per_question=1,
        seg_token_num=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False,
    ):
        self.reason_seg_data = reason_seg_data
        self.seg_token_num = seg_token_num
        self.num_classes_per_sample = num_classes_per_sample
        self.samples_per_epoch = samples_per_epoch
        self.base_image_dir = base_image_dir

        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.long_question_list = LONG_QUESTION_TEMPLATE
        self.answer_list = LONG_ANSWER_TEMPLATE

        self.masks_process_with_clip = masks_process_with_clip
        self.pad_train_clip_images = pad_train_clip_images
        self.transform = ResizeLongestSide(image_size)  # for SAM
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])
        
        # load dataset
        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))
    
    def load_and_preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
        sam_input_shape = tuple(image.shape[:2])
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        
        return image, image_clip, sam_input_shape
    
    def __len__(self):
        return self.samples_per_epoch
    
    def preprocess(self, x: torch.Tensor, decoder_image_size) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = decoder_image_size - h
        padw = decoder_image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)

        mode_this_turn = random.choices(self.choices, self.weights, k=1)[0]
        image_path = images[idx]
        json_path = jsons[idx]
        
        # Load images and clip features
        # image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        # preprocess image for clip
        if self.pad_train_clip_images:
            image_clip = self.transform_clip.apply_image(image)
            clip_resize = image_clip.shape[:2]
            # print("self.clip_image_processor.size['shortest_edge']:", self.clip_image_processor.size['shortest_edge'])
            image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_image_processor.size['shortest_edge'])
        else:
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            clip_resize = image_clip.shape[-2:]

        # Get sents and segmentation maps
        # img = cv2.imread(image_path)[:, :, ::-1]
        mask, sents, fp_qa, is_sentence = get_mask_from_json_v2(json_path, image)
        # Sampling
        sample_size = min(len(sents), self.num_classes_per_sample)
        sampled_inds = random.sample(range(len(sents)), sample_size) if len(sents) >= self.num_classes_per_sample else range(len(sents))
        neg_sample_size = min(len(fp_qa), self.num_classes_per_sample)
        neg_sampled_inds = random.sample(range(len(fp_qa)), neg_sample_size) if len(fp_qa) >= self.num_classes_per_sample else range(len(fp_qa))
        
        sampled_sents = [sents[idx] for idx in sampled_inds]
        neg_sampled_sents = [fp_qa[idx] for idx in neg_sampled_inds]
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        # Create Q/A Data
        questions = []
        answers = []
        conversations = []
        for idx, text in enumerate(sampled_sents):
            neg_text = neg_sampled_sents[idx]
            if mode_this_turn == "True_Premise":
                if is_sentence:
                    question_template = random.choice(self.long_question_list)
                    questions.append(question_template.format(sent=text))
                else:
                    question_template = random.choice(self.short_question_list)
                    questions.append(question_template.format(class_name=text.lower()))
                answers.append(random.choice(self.answer_list))
            else:
                if neg_text[1] is True:
                    question_template = random.choice(self.long_question_list)
                    questions.append(question_template.format(sent=neg_text[0]))
                else:
                    question_template = random.choice(self.short_question_list)
                    questions.append(question_template.format(class_name=neg_text[0]))
                answers.append(neg_text[2])
                
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], questions[idx])
            conv.append_message(conv.roles[1], answers[idx])
            conversations.append(conv.get_prompt())

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)
  
        if mode_this_turn != "True_Premise":
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
        else:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks.astype(np.uint8))
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        seg_count = ' '.join(conversations).count('[SEG') / self.seg_token_num
        valid_masks = sum(1 for j in range(masks.shape[0]) if masks[j].sum() > 0)
        if valid_masks != seg_count: return self.__getitem__(0)

        if self.masks_process_with_clip:
            mask_shape =  image_clip.shape[-1]
            if len(masks) == 0:
                masks = torch.zeros(0, mask_shape, mask_shape)
            else:
                masks = transform_mask(masks, mask_shape)
        
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize, # input for sam
            clip_resize,
            questions,
            sampled_sents,
            False,  # use_assign_list
            False   # inference
        )

def transform_mask(masks, size):
    height, width = masks.shape[-2:]
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    new_shape = (new_long, new_short) if width <= height else (new_short, new_long)
    masks = F.interpolate(masks[None].float(), size=new_shape, mode="nearest")[0].bool()

    orig_height, orig_width = new_shape
    crop_height, crop_width = size, size
    crop_height, crop_width = int(crop_height), int(crop_width)
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (orig_width - crop_width) // 2
    right = left + crop_width
    assert top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width
    # if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
    masks = masks[..., top:bottom, left:right]

    return masks
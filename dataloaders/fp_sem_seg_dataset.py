import glob
import json
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide

from model.llava import conversation as conversation_lib
from .qa_template import NEG_ANSWER_TEMPLATE, CORRECT_ANSWER_TEMPLATE, SHORT_QUESTION_TEMPLATE, SHORT_ANSWER_TEMPLATE

def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    with open("dataloaders/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ade20k/images", "training"))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "ade20k",
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir):
    cocostuff_classes = []
    with open("dataloaders/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "cocostuff", "train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


class fpSemSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    # Define your list of choices and their corresponding weights
    choices = ["True_Premise", "False_Premise_Denial", "False_Premise_Correction"]  
    weights = [0.85, 0.07, 0.08]
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
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        num_classes_per_question=1,
        seg_token_num=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False,
    ):
        self.pad_train_clip_images = pad_train_clip_images
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.seg_token_num = seg_token_num
        self.base_image_dir = base_image_dir

        # Image transformations
        self.transform = ResizeLongestSide(image_size)  # for SAM
        self.masks_process_with_clip = masks_process_with_clip
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])  # for CLIP


        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.answer_list = SHORT_ANSWER_TEMPLATE
        self.neg_answer_list = NEG_ANSWER_TEMPLATE
        self.correct_answer_list = CORRECT_ANSWER_TEMPLATE

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes
            # print(f"{ds}: {self.data2classes[ds]}", flush=True)
        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }
    
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

    def select_samples(self, annotations):
        """Select a given number of samples from annotations."""
        if len(annotations) >= self.num_classes_per_sample:
            return random.sample(annotations, self.num_classes_per_sample)
        return annotations

    def generate_class_names(self, annotations, class_map):
        """Generate class names from annotations."""
        class_names = []
        for ann in annotations:
            category_id = ann if isinstance(ann, int) else ann["category_id"]
            sampled_cls = class_map[category_id]
            if isinstance(sampled_cls, tuple):
                obj, part = sampled_cls
                name = f"{obj} {part}" if random.random() < 0.5 else f"the {part} of the {obj}"
            else:
                name = sampled_cls
            class_names.append(name)
        return class_names
    
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

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]

        mode_this_turn = random.choices(self.choices, self.weights, k=1)[0]
        if ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            # Sample one image
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            # construct_image_path
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
            
            # Load and preprocess image 
            # image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_path)
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]

            # Load annotations
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            # Positive Samples
            anns = coco_api.loadAnns(annIds)
            # re-sample if this sample does not contain annotations
            if len(anns) == 0:
                return self.__getitem__(0)
            # Negative Samples
            all_cats = {cat["id"] for cat in coco_api.cats.values()}
            neg_anns = list(set(all_cats) - set([ann["category_id"] for ann in anns]))

            sampled_anns = self.select_samples(anns)
            neg_sampled_anns = self.select_samples(neg_anns)
            sampled_classes = self.generate_class_names(sampled_anns, class_map)
            neg_sampled_classes = self.generate_class_names(neg_sampled_anns, class_map)

        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            image, labels = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            label_path = labels[idx]
            label = Image.open(label_path)
            label = np.array(label)
            if ds == "ade20k":
                label[label == 0] = self.ignore_label
                label -= 1
                label[label == 254] = self.ignore_label
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = self.ignore_label
            
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]

            # image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_path)
            unique_label = np.unique(label).tolist()
            if self.ignore_label in unique_label:
                unique_label.remove(self.ignore_label)
            if len(unique_label) == 0:
                return self.__getitem__(0)
            all_labels = [x for x in range(len(self.data2classes[ds]))]
            neg_unique_label = list(set(all_labels) - set(unique_label))

            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            neg_classes = [
                self.data2classes[ds][class_id] for class_id in neg_unique_label
            ]
            sampled_classes = self.select_samples(classes)
            neg_sampled_classes = self.select_samples(neg_classes)

        # conversation + QAs
        class_ids = []
        questions = []
        answers = []
        conversations = []
        seg_token = ["[SEG{}]".format(i) for i in range(self.seg_token_num)]
        seg_token = ' '.join(seg_token)
        
        for idx, sampled_cls in enumerate(sampled_classes):
            text = sampled_cls
            neg_text = neg_sampled_classes[idx]
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)

            if mode_this_turn == "True_Premise":
                questions.append(question_template.format(class_name=text.lower()))
                answer_template = random.choice(self.answer_list)
                answers.append(answer_template.format(class_name=text.lower()))
            elif mode_this_turn == "False_Premise_Denial":
                questions.append(question_template.format(class_name=neg_text.lower()))
                answer_template = random.choice(self.neg_answer_list)
                answers.append(answer_template.format(class_name=neg_text.lower()))
            elif mode_this_turn == "False_Premise_Correction":
                questions.append(question_template.format(class_name=neg_text.lower()))
                answer_template = random.choice(self.correct_answer_list)
                answers.append(
                    answer_template.format(
                        class_name=neg_text.lower(), gt_name=text.lower()
                    )
                )
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], questions[idx])
            conv.append_message(conv.roles[1], answers[idx])
            conversations.append(conv.get_prompt())
            if ds in ["ade20k", "cocostuff", "mapillary"]:
                class_id = self.data2classes[ds].tolist().index(sampled_cls)
                class_ids.append(class_id)
      
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)

        # Load annotations
        if ds in ["paco_lvis", "pascal_part"]:
            masks = []
            for ann in sampled_anns:
                try:
                    masks.append(coco_api.annToMask(ann))
                except Exception as e:
                    print(e)
                    return self.__getitem__(0)

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks.astype(np.uint8))
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        else:
            label = torch.from_numpy(label).long()
            masks = []
            for class_id in class_ids:
                masks.append(label == class_id)
            masks = torch.stack(masks, dim=0)
        
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
            sampled_classes,
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
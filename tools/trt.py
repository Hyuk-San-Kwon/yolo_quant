#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import shutil
from loguru import logger

import tensorrt as trt
import torch
import PIL
from PIL import Image
import torchvision.transforms as transforms
tf = transforms.ToTensor()
from torch2trt import torch2trt
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from yolox.data.data_augment import ValTransform
import cv2
from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX ncnn deploy")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "-w", '--workspace', type=int, default=32, help='max workspace size in detect'
    )
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')
    parser.add_argument("-t", '--type', type=str, default=1, help='fp16 or int8')
    return parser


@logger.catch
@torch.no_grad()
def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
   
    model = exp.get_model()
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    transform=Compose([
                Resize((416, 416)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    model.eval()
    model.cuda()
    model.head.decode_in_inference = False
    logger.info("cuda loading done.")
    x = torch.randn(8, 3, exp.test_size[0], exp.test_size[1]).cuda()
    
    if args.type == "int8":
        
        path = "./assets/VOCdevkit/VOC2012/JPEGImages"
        file_list = os.listdir(path)
        img = cv2.imread(path + '/' + file_list[0])
        preproc = ValTransform(legacy=False)
        img, _ = preproc(img, None, (416,416))
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        result = img
        
        for name in file_list[:100]:
            img = cv2.imread(path + '/' + name)
            img, _ = preproc(img, None, (416,416))
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.float()
            result = torch.cat((result, img))
            
        logger.info(result.size())
        
        result = result.cuda()
        model_trt = torch2trt(
            model,
            [x],
            int8_mode = True,
            log_level=trt.Logger.INFO,
            int8_calib_batch_size=32,
            int8_calib_dataset=[result],
            # max_workspace_size=(1 << args.workspace),
            # max_batch_size=32,
        
        )
    # fp16
    else:
    
        model_trt = torch2trt(
            model,
            [x],
            fp16_mode = True,
            log_level=trt.Logger.INFO,
            max_workspace_size=(1 << args.workspace),
            max_batch_size=32,
        
        )
    torch.save(model_trt.state_dict(), os.path.join(file_name, args.type + ".pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join(file_name, "model_trt.engine")
    engine_file_demo = os.path.join("demo", "TensorRT", "cpp", args.type + "_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    shutil.copyfile(engine_file, engine_file_demo)

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()

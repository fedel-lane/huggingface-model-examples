#!/usr/bin/env python
# Use Diffusion Pipeline to generate an image from a description
# Example:
# bash$ python viscpm-paint.py --out /tmp/t.png A photo of dogs playing poker
#  Loading tokenizer
#  Loading encoder
#  Loading diffusion pipeline
#  Generating image from:
#  A photo of dogs playing poker
#  Generated 1 images
#  Saving image to /tmp/t.png  

import os
import argparse

import sys
from transformers import AutoModel, AutoTokenizer
from diffusers import DiffusionPipeline
# NOTE: model's pipeline.py needs this added:
# from diffusers.utils.torch_utils import randn_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Chinese-English bilingual multi-modal large model series based on CPM (Chinese Pretrained Models) basic model VisCPM is a family of open-source large multimodal models, which support multimodal conversational capabilities (VisCPM-Chat model) and text-to-image generation capabilities (VisCPM-Paint model) in both Chinese and English, achieving state-of-the-art peformance among Chinese open-source multimodal models. VisCPM is trained based on the large language model CPM-Bee with 10B parameters, fusing visual encoder (Q-Former) and visual decoder (Diffusion-UNet) to support visual inputs and outputs. Thanks to the good bilingual capability of CPM-Bee, VisCPM can be pre-trained with English multimodal data only and well generalize to achieve promising Chinese multimodal capabilities. VisCPM-Paint supports bilingual text-to-image generation. The model uses CPM-Bee as the text encoder, UNet as the image decoder, and fuses vision and language models using the objective of diffusion model. During the training process, the parameters of the language model remain fixed. The visual decoder is initialized with the parameters of Stable Diffusion 2.1, and it is fused with the language model by gradually unfreezing key bridging parameters. The model is trained on the LAION 2B English text-image pair dataset. Similar to VisCPM-Chat, we found that due to the bilingual capability of CPM-Bee, VisCPM-Paint can achieve good Chinese text-to-image generation by training only on English text-image pairs, surpassing the performance of Chinese open-source models. By incorporating an additional 20M cleaned native Chinese text-image pairs and 120M translated text-image pairs in Chinese, the model's Chinese text-to-image generation ability can be further improved. We sample 30,000 images from the standard image generation test set MSCOCO and calculated commonly used evaluation metrics FID (Fréchet Inception Distance) to assess the quality of generated images. Similarly, we provide two versions of the model, namely VisCPM-Paint-balance and VisCPM-Paint-zhplus. The former has a balanced ability in both English and Chinese, while the latter emphasizes Chinese proficiency. VisCPM-Paint-balance is trained only using English text-image pairs, while VisCPM-Paint-zhplus incorporates an additional 20M native Chinese text-image pairs and 120M translated text-image pairs in Chinese based on VisCPM-Paint-balance.",
            epilog="")
    parser.add_argument('--out', dest='out_file', type=argparse.FileType('wb'),
                        help='Name of PNG file to write')
    parser.add_argument('description', type=str, nargs='+',
                        help='Description, e.g. ""')
    args = parser.parse_args()

    model_name = 'openbmb/VisCPM-Paint'


    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Loading encoder")
    text_encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    print("Loading diffusion pipeline")
    pipeline = DiffusionPipeline.from_pretrained(model_name, custom_pipeline=model_name, text_encoder=text_encoder, tokenizer=tokenizer)


    description = " ".join(args.description)
    print("Generating image from: \n%s" % description)
    result = pipeline(description)
    print("Generated %d images" % len(result.images))
    image = result.images[0]

    print("Saving image to %s" % args.out_file.name)
    image.save(args.out_file)


#!/usr/bin/env python
# (c) Copyright 2023 Fedel Lane Associates

import os
import sys
import argparse
import readline
import time
from huggingface_hub import InferenceClient

# ----------------------------------------------------------------------
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

# ----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
            description='HuggingFace Inference API for image generation',
            epilog="Example: image_generation.py The road was long and")

    parser.add_argument('-i', '--interactive', dest='interactive', action='store_true',
                    help='Use interactive prompt look')
    parser.add_argument('--token', dest='token', default=None, required=False,
                    help='HuggingFace API Token')
    parser.add_argument('--model', dest='model', default=None, required=False,
                    help='Override the default model for the text_generation task')
    parser.add_argument('-o', '--output-file', dest='output_file', default=None, 
                    type=str, required=False,
                    help='Output file (not required if using -i)')
    parser.add_argument('--height', dest='height', default=None, type=float,
                    required=False,
                    help='The height in pixels of the image to generate')
    parser.add_argument('--width', dest='width', default=None, type=float,
                    required=False,
                    help='The width in pixels of the image to generate')
    parser.add_argument('--num-inference', dest='num_inference', 
                    default=None, type=float, required=False,
                    help='The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.')
    parser.add_argument('--guidance-scale', dest='guidance_scale', 
                    default=None, type=float, required=False,
                    help='Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.')
    parser.add_argument('--negative-prompt', dest='negative_prompt', 
                    default='nsfw, nude, censored, bad anatomy, bad hands, extra digits, cropped, jpeg artifacts, error', 
                    help='List elements to not exclude from image')
    parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Enable debug output')
    parser.add_argument('--time', dest='time', action='store_true',
                    help='Time model calls')
    parser.add_argument('text_input', nargs='*',
                        help='Text to use as input')
    
    args = parser.parse_args()
    return args

# ----------------------------------------------------------------------

def perform_task(text_input, token, model, args):
    client = InferenceClient(token=token)
    with Timer() as t:
        result =  client.text_to_image(text_input,
                                       negative_prompt=args.negative_prompt,
                                       height=args.height,
                                       width=args.width,
                                       num_inference_steps=args.num_inference,
                                       guidance_scale=args.guidance_scale,
                                       model=model)
    if args.time: print("[TIME] Model response: %.03f sec" % t.interval)
    return result

def perform_and_save(text_input, token, model, args):
    try:
        with Timer() as t:
            img = perform_task(text_input, token, model, args)
    finally:
        print("> Image geneated (%.03f sec). Enter filename to write to:" % t.interval)
    line = input()
    try:
        img.save(line.strip())
    except:
        print("Could not write to file %s! Discarding." % line)

def enter_loop(orig_input, token, args):
    # =================
    # Enter prompt loop
    while True:
        try:
            if len(orig_input) > 0:
                perform_and_save(orig_input, token, args.model, args)

            print("> Enter description of image to generate:")
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        if line.strip() == "":
            continue
        perform_and_save(line, token, args.model, args)

def get_token(args):
    api_key = None
    if args.token:
        api_key = token
    elif "HUGGINGFACEHUB_API_TOKEN" in os.environ:
        api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    else:
        try:
            api_key = os.popen('huggingface_key').read()
        except:
            api_key = None
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    return api_key

def enforce_loaded_model(args):
    # check model
    if args.model:
        stat = InferenceClient(token=token).get_model_status(args.model)
        if not stat.loaded:
            print("Unable to access model %s: not loaded" % args.model)
            print(str(stat))
            sys.exit()

# ----------------------------------------------------------------------
if __name__ == '__main__':
    args = get_args()
    token = get_token(args)
    enforce_loaded_model(args)

    text_input = ' '.join(args.text_input)
    if args.interactive:
        enter_loop(text_input, token, args)
    else:
        img = perform_task(text_input, token, args.model, args)
        img.save(args.output_file)

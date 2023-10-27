#!/usr/bin/env python
# (c) Copyright 2023 Fedel Lane Associates
# Example:
#bash$ python inference_client/converational.py 
#> why is the sky blue?
#>>  The sky is blue because of an optical effect known as Rayleigh scattering.
#> Can you explain that in plain English?
#>>  It is a phenomenon that occurs when light is scattered by particles with a wavelength between approximately 450 and 495 nanometres.

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
            description='HuggingFace Inference API for conversational',
            epilog="Example: conversational.py 'Why is the sky blue?'")

    parser.add_argument('--token', dest='token', default=None, required=False,
                    help='HuggingFace API Token')
    parser.add_argument('--model', dest='model', default=None, required=False,
                    help='Model to use, e.g. "microsoft/DialoGPT-large"')
    parser.add_argument('--max-length', dest='max_length', default=None,
                    type=int, help='Max tokens to generate ')
    parser.add_argument('--min-length', dest='min_length', default=None,
                    type=int, help='Min tokens to generate ')
    parser.add_argument('--max-time', dest='max_time', default=None,
                    type=int, help='Max time in seconds a query can take')
    parser.add_argument('--repetition-penalty', 
                    dest='repetition_penalty', default=None, type=float,
                    help='Penalize repeated tokens to reduce occurrence')
    parser.add_argument('--temperature', dest='temperature', default=0.6,
                    help='Temperature (creativity) of model [0.6]')
    parser.add_argument('--top-k', dest='top_k', default=10,
                    help='# of top labels returned by the pipeline [10]')
    parser.add_argument('--top-p', dest='top_p', default=0.95,
                    help='Probability set of words must exceed [0.95]')
    parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Enable debug output')
    parser.add_argument('--time', dest='time', action='store_true',
                    help='Time model calls')
    parser.add_argument('text_input', nargs='*',
                        help='Input')
    
    args = parser.parse_args()
    return args

def build_params(args):
    params = {}
    if args.min_length:
        params['min_length'] = args.min_length
    if args.max_length:
        params['max_length'] = args.max_length
    if args.top_k:
        params['top_k'] = args.top_k
    if args.top_p:
        params['top_p'] = args.top_p
    if args.temperature:
        params['temperature'] = args.temperature
    if args.repetition_penalty:
        params['repetition_penalty'] = args.repetition_penalty
    if args.max_time:
        params['max_time'] = args.max_time

    return params

# ----------------------------------------------------------------------
def perform_task(text_input, hist_input, hist_resp, token, model, params):
    client = InferenceClient(token=token)

    with Timer() as t:
        result = client.conversational(text_input, 
                                       hist_resp, 
                                       hist_input, 
                                       parameters=params, 
                                       model=model)
    if args.time: print("[TIME] Model response: %.03f sec" % t.interval)
    return result 

def enter_loop(orig_input, token, args):
    past_inputs = []
    past_responses = []
    params = build_params(args)

    # =================
    # Enter prompt loop
    while True: 
        try:
            if len(orig_input) > 0:
                rv = perform_task(line, [], [], token, args.model, params)
                reply = rv['generated_text']
                past_inputs.append(orig_input)
                past_responses.append(reply)
                print(">> " + reply)

            print("> ", end="")
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        if line.strip() == "":
            continue

        try:
            with Timer() as t:
                rv = perform_task(line, past_inputs, past_responses, token, args.model, params)
            reply = rv['generated_text']
            past_inputs.append(line)
            past_responses.append(reply)
            print("\n>> " + reply)
        finally:
            if args.time:
                print("> (Elapsed: %.03f sec)" % t.interval)

    return ""

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
    enter_loop(text_input, token, args)


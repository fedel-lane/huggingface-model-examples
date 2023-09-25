#!/usr/bin/env python
# Example:
# python codegen-completion.py "def hello():"

import os
import sys
import time
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


def get_args():
    parser = argparse.ArgumentParser(
            description='CodeGen is a family of open-source model for program synthesis. For regular causal sampling, simply generate completions given the context: text = "def hello_world():"',
            epilog="")
    parser.add_argument('--in-file', dest='in_file', 
                        type=argparse.FileType('r'),
                        help='File to read input from')
    parser.add_argument('--out-file', dest='out_file', 
                        type=argparse.FileType('w'),
                        help='File to write output to')
    parser.add_argument('--model', dest='model', default = "codegen25-7b-multi",
                        help='codegen2-7b-mono codegen2-16n codegen2-1B-multi cdegen25-7b-instruct [codegen25-7b-multi]')
    parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Enable debug output')
    parser.add_argument('--time', dest='time', action='store_true',
                    help='Time model calls')
    parser.add_argument('code', type=str, nargs='*',
                    help='Code to complete, e.g. "def connect():"')


    args = parser.parse_args()
    return args

# ----------------------------------------------------------------------
if __name__ == '__main__':
    args = get_args()    

    model_name = "Salesforce/" + args.model

    code = ""
    if args.in_file:
        code = args.in_file.read()
    code = code + " ".join(args.code)

    if len(code) <= 1:
        print("Nothing to do! Please specify some candidate code")
        sys.exit()


    if args.debug: print("[INFO] Loading Tokenizer")
    with Timer() as t:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if args.time: print("[TIME] Model instantiation: %.03f sec" % t.interval)

    if args.debug: print("[INFO] Loading Model")
    with Timer() as t:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    if args.time: print("[TIME] Model instantiation: %.03f sec" % t.interval)


    if args.debug: print("[INFO] Tokenizing input code")
    input_ids = tokenizer(code, return_tensors="pt").input_ids

    if args.debug: print("[INFO] Generating output code")
    generated_ids = model.generate(input_ids, max_length=128)

    if args.debug: print("[INFO] Generated %d output candidates" % len(generated_ids))
    if args.debug: print("[INFO] Decoding %d output tokens" % len(generated_ids[0]))
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    if args.out_file:
        args.out_file.write(output)
    else:
        print(output)

#!/usr/bin/env python
# 
# From the docs:
# def format(prefix, suffix):
#   return prefix + "<mask_1>" + suffix + "<|endoftext|>" + "<sep>" + "<mask_1>"
# prefix = "def hello_world():\n    "
# suffix = "    return name"
# code = format(prefix, suffix)

import os
import sys
import argparse
import time

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
            description="CodeGen2 is a family of autoregressive language models for program synthesis, introduced in the paper: CodeGen2: Lessons for Training LLMs on Programming and Natural Languages by Erik Nijkamp*, Hiroaki Hayashi*, Caiming Xiong, Silvio Savarese, Yingbo Zhou.  CodeGen2 was trained using cross-entropy loss to maximize the likelihood of sequential inputs. The input sequences are formatted in two ways: (1) causal language modeling and (2) file-level span corruption. Please refer to the paper for more details.",
            epilog=""""
For infill sampling, we introduce three new special token types:
    <mask_N>: N-th span to be masked. In practice, use <mask_1> to where you want to sample infill.
    <sep>: Separator token between the suffix and the infilled sample. See below.
    <eom>: "End-Of-Mask" token that model will output at the end of infilling. You may use this token to truncate the output.
For example, if we want to generate infill for the following cursor position of a function:
def hello_world():
    |
    return name
we construct an input to the model by
    Inserting <mask_1> token in place of cursor position
    Append <sep> token to indicate the boundary
    Insert another <mask_1> to indicate which mask we want to infill.
    """ )           
    parser.add_argument('--in-file', dest='in_file', 
                        type=argparse.FileType('r'),
                        help='File to read input from. Must have <mask_1>')
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
                    help='Code to complete, e.g. "def connect():". Must have <mask_1>.')


    args = parser.parse_args()
    return args

MASK1_TOK = '<mask_1>' # use <mask_1> to where you want to sample infill.
SEP_TOK = '<sep>'      # use between the suffix and the infilled sample.
END_TOK = '<eom>'      # token that model will output at the end of infilling.
EOT_TOK = '<|endoftext|>'

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

    # if code has no mask tok, nothing to do!
    if MASK1_TOK not in code:
        print("No %s in input! Adding at end." % MASK1_TOK)
        code = code + MASK1_TOK

    text = code + EOT_TOK + SEP_TOK + MASK1_TOK
    if args.debug: print("[INFO] Using transformed code:\n\t%s" % text)

    if args.debug: print("[INFO] Loading Tokenizer")
    with Timer() as t:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if args.time: print("[TIME] Model instantiation: %.03f sec" % t.interval)

    if args.debug: print("[INFO] Loading Model")
    with Timer() as t:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    if args.time: print("[TIME] Model instantiation: %.03f sec" % t.interval)



    if args.debug: print("[INFO] Tokenizing input code")
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    if args.debug: print("[INFO] Generating output code")
    generated_ids = model.generate(input_ids, max_length=128)

    if args.debug: print("[INFO] Generated %d output candidates" % len(generated_ids))
    if args.debug: print("[INFO] Decoding %d output tokens" % len(generated_ids[0]))
    # FIXME: Look for END_TOK
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)[len(text):]

    if args.out_file:
        args.out_file.write(output)
    else:
        print(output)


#!/usr/bin/env python
# A simple example of calling the HF text-generation pipeline directly on
# an Alpaca (instruction-following LLAMA) model

import os
import argparse
import readline
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
            description='Chatbot based on alpaca')
    parser.add_argument('--no-sample', dest='sample', action='store_false',
                    help='Disable decoding strategies like top-p top-k etc')
    parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=1250,
                    help='Max length generated content [1250]')
    parser.add_argument('--num-beams', dest='num_beams', default=1,
                    help='If > 1, use beam search instead of greedy [1]')
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
    
    args = parser.parse_args()
    return args

# ----------------------------------------------------------------------
if __name__ == '__main__':
    args = get_args()
    model_name="chavinlo/alpaca-native"
    
    # ==============
    # create the LLM
    if args.debug: print("[INFO] Instantiating tokenizer: %s" % model_name)
    with Timer() as t:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.time: print("[TIME] Tokenizer instantiation: %.03f sec" % t.interval)

    if args.debug: print("[INFO] Instantiating model: %s" % model_name)
    with Timer() as t:
        pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens, 
            do_sample=args.sample,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_return_sequences=1,
            num_beams=args.num_beams,
            early_stopping=(args.num_beams > 1),
            eos_token_id=tokenizer.eos_token_id,
            trust_remote_code=True,
        )
    if args.time: print("[TIME] Model instantiation: %.03f sec" % t.interval)

    # =================
    # Enter prompt loop
    while True:
        try:
            print("> Enter some text to complete:")
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        if line.strip() == "":
            continue

        try:
            with Timer() as t:
                rv = pipe(line)
            response = rv[0]['generated_text']
            print(">> " + response)
        finally:
            if args.time:
                print("> (Elapsed: %.03f sec)" % t.interval)


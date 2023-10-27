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
            description='HuggingFace Inference API for text generation',
            epilog="Example: text_generation.py --temperature 0.9 The road was long and")

    parser.add_argument('-i', '--interactive', dest='interactive', action='store_true',
                    help='Use interactive prompt look')
    parser.add_argument('--token', dest='token', default=None, required=False,
                    help='HuggingFace API Token')
    parser.add_argument('--model', dest='model', default=None, required=False,
                    help='Override the default model for the text_generation task')
    parser.add_argument('--no-sample', dest='sample', action='store_false',
                    help='Disable decoding strategies like top-p top-k etc')
    parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=1250,
                    help='Max length generated content [1250]')
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
                        help='Text to use as input (overrides --source)')
    
    args = parser.parse_args()
    return args

# ----------------------------------------------------------------------

def perform_task(text_input, token, model, args):
    client = InferenceClient(token=token)
    with Timer() as t:
        result =  client.text_generation(text_input, model=model,
                                         do_sample=args.sample,
                                         max_new_tokens=args.max_new_tokens,
                                         temperature=args.temperature,
                                         top_k=args.top_k,
                                         top_p=args.top_p,
                                         #truncate [int] = None
                                         #typical_p [float] = None
                                         #watermark bool = False
                                         #decoder_input_details [bool] = False
                                         #best_of [int] = None
                                         #repetition_penalty [float] = None
                                         #return_full_text [bool] = False
                                         #seed [int] = None
                                         #stop_sequences: [List[str]] = None
                )
    if args.time: print("[TIME] Model response: %.03f sec" % t.interval)
    return result

def enter_loop(orig_input, token, args):
    # =================
    # Enter prompt loop
    while True:
        try:
            if len(orig_input) > 0:
                print(perform_task(orig_input, token, args.model, args))

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
                rv = perform_task(line, token, args.model, args)
            print("> " + rv)
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
    if len(args.text_input) < 1 and args.source:
        with open(args.source) as f:
            text_input = f.read()

    if args.interactive:
        enter_loop(text_input, token, args)
    else:
        print(perform_task(text_input, token, args.model, args))

    sys.exit()
    #with Timer() as t:
    #    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #if args.time: print("[TIME] Tokenizer instantiation: %.03f sec" % t.interval)

    #if args.debug: print("[INFO] Creating Langchain HuggingFacePipeline")

    # =====================
    # Create Prompt + Chain
    template = """Question: {question}

    Answer: Let's think step by step."""
    if args.debug: print("[INFO] Creating prompt template from:\n%s" % template)
    prompt = PromptTemplate(template=template, input_variables=["question"])

    if args.debug: print("[INFO] Creating Langchain LLMChain")
    llm_chain = LLMChain(prompt=prompt, llm=llm)

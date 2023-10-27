#!/usr/bin/env python
# Test code to get up-and-running with a downloaded AgentLM model, one of:
#   https://huggingface.co/THUDM/agentlm-7b
#   https://huggingface.co/THUDM/agentlm-13b
#   https://huggingface.co/THUDM/agentlm-70b
# Example:
# bash$ python agentlm.py

import os
import argparse
import readline
import time

from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import LlamaTokenizer

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
            description='',
            epilog="")
    parser.add_argument('--small', dest='small_model', action='store_true',
                    help='Use smallest model available locally')
    parser.add_argument('--large', dest='large_model', action='store_true',
                    help='Use largest model available locally')
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

    model_name = 'THUDM/agentlm-'
    if args.small_model:
        model_name = model_name + '7b'
    elif args.large_model:
        model_name = model_name + '70b'
    else:
        model_name = model_name + '13b'

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
            max_new_tokens=args.max_new_tokens, # replaces max_length
            do_sample=args.sample,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_return_sequences=1, # just support 1 response in prompt-loop
            num_beams=args.num_beams,
            early_stopping=(args.num_beams > 1),
            eos_token_id=tokenizer.eos_token_id,
            trust_remote_code=True,
            #torch_dtype=torch.bfloat16,
            #device_map="auto",
        )
    if args.time: print("[TIME] Model instantiation: %.03f sec" % t.interval)

    if args.debug: print("[INFO] Creating Langchain HuggingFacePipeline")
    llm = HuggingFacePipeline(pipeline=pipe)

    # =====================
    # Create Prompt + Chain
    template = """Question: {question}

    Answer: Let's think step by step."""
    if args.debug: print("[INFO] Creating prompt template from:\n%s" % template)
    prompt = PromptTemplate(template=template, input_variables=["question"])

    if args.debug: print("[INFO] Creating Langchain LLMChain")
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print("> Using Q&A prompt with cue \"Let's think step by step.\" Model: %s" % model_name)

    # =================
    # Enter prompt loop
    while True:
        try:
            print("> Ask a question:")
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        if line.strip() == "":
            continue

        try:
            with Timer() as t:
                rv = llm_chain.run(line)
            print("> " + rv)
        finally:
            if args.time:
                print("> (Elapsed: %.03f sec)" % t.interval)


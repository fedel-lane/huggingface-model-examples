#!/usr/bin/env python
# Test code to get up-and-running with a downloaded Microsoft Ph-2 model:
#   https://huggingface.co/microsoft/phi-2
# Example:

import os
import argparse
import readline
import time

from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, pipeline

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
    parser.add_argument('--qa', dest='qa_mode', action='store_true',
            help='Prompt templates is "Instruct:/Output:"')
    parser.add_argument('--chat', dest='chat_mode', action='store_true',
            help='Prompt templates is "User:/Agent:"')
    parser.add_argument('--code', dest='code_mode', action='store_true',
            help='Prompt templates is verbatim code')
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

    model_name = 'microsoft/phi-2'

    # ==============
    # create the LLM
    #if args.debug: print("[INFO] Instantiating tokenizer: %s" % model_name)
    with Timer() as t:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    #if args.time: print("[TIME] Tokenizer instantiation: %.03f sec" % t.interval)

    if args.debug: print("[INFO] Instantiating model: %s" % model_name)
    with Timer() as t:
        #pipe = pipeline("text-generation", 
        #                model="microsoft/phi-2",
        #                max_new_tokens=args.max_new_tokens,
        #                )
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
            return_full_text=False
            #trust_remote_code=True,
            #torch_dtype=torch.bfloat16,
            #device_map="auto",
        )
    if args.time: print("[TIME] Model instantiation: %.03f sec" % t.interval)

    if args.debug: print("[INFO] Creating Langchain HuggingFacePipeline")

    # =====================
    # Create Prompt + Chain
    if args.qa_mode:
        template = """Instruct: {question}
Output: Let's think step by step."""
    elif args.chat_mode:
        template = """User: {question}
Agent:"""
    elif args.code_mode:
        template = """{question}
"""
    else:
        template = """{question}
"""
    if args.debug: print("[INFO] Creating prompt template from:\n%s" % template)
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # =================
    # Enter prompt loop
    memory = ConversationBufferMemory()
    print("> Ask a question:")
    while True:
        try:
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
                query = prompt.format(question=line)
                ctx = memory.load_memory_variables({})
                if len(ctx) > 0:
                    query = ctx['history'] + "\n" + query

                rv = pipe(query)
            print("vvvvvvv")
            print(rv)
            print("^^^^^^^")
            if args.chat_mode:
                resp = rv[0]['generated_text'].strip()
                #resp = rv[0]['generated_text'].lstrip("User: " + line).strip()
                resp = resp.split("\n")[0]
                memory.save_context({"User": line}, {"Agent": resp})
            else:
                resp = rv[0]['generated_text'].split("\n\n")[0]
                #resp = rv[0]['generated_text'].lstrip(line).strip()
                
            print("> " + resp)

        finally:
            if args.time:
                print("> (Elapsed: %.03f sec)" % t.interval)


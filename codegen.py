#!/usr/bin/env python
# Generate code from a natural language
# Example:
# python codegen-completion.py "def hello():"
# def hello():
#     return "Hello World!"
# 
# @app.route('/')
# def index():
#     return render_template('index.html')


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

SEP_TOK = '<sep>'      # use between the suffix and the infilled sample.
END_TOK = '<eom>'      # token that model will output at the end of infilling.
EOT_TOK = '<|endoftext|>'

def get_args():
    default_prompt="#/usr/bin/python\nimport os"

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
    parser.add_argument("--prompt", dest="prompt", default=default_prompt,
                    help="Prompt to prefix code description with"),
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

    specs = []
    code = ""
    if args.in_file:
        for line in args.in_file.readlines():
            if line.strip() == "":
                continue
            specs.append(line)
    if len(args.code) > 0:
        specs.append(" ".join(args.code).strip())

    if len(specs) < 1:
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


    output_code = []
    for idx, spec in enumerate(specs):
        # FIXME: This is only for python!
        #text = args.prompt + "\n'''" + spec + "\n'''"
        text = "'''" + spec + "\n'''"

        if args.debug: print("[INFO] Tokenizing input code")
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        if args.debug: print("[INFO] Generating output code")
        generated_ids = model.generate(input_ids, min_length=16,
                                       max_length=2048,
                                       #temperature=0.6, 
                                       #repetition_penalty=1.3,
                                       pad_token_id=tokenizer.eos_token_id)

        if args.debug: print("[INFO] Generated %d output candidates" % len(generated_ids))
        if args.debug: print("[INFO] Decoding %d output tokens" % len(generated_ids[0]))
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        #print("[%d] : %s" % (idx, output))
        output_code.append(output.split(END_TOK)[0])

    print("Generated %d entries" % len(output_code))
    output = "\n\n".join(output_code)
    if args.out_file:
        args.out_file.write(output)
    else:
        print(output)

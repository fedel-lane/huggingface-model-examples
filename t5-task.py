#!/usr/bin/env python
# FROM:
# https://medium.com/artificialis/t5-for-text-summarization-in-7-lines-of-code-b665c9e40771

import os
import argparse
import readline
import time

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

def get_args():
    parser = argparse.ArgumentParser(
            description='',
            epilog="")
    parser.add_argument('--flan', dest='flan_model', action='store_true',
                    help='Use flan-t5 instead of original t5')
    parser.add_argument('--large', dest='large_model', action='store_true',
                    help='Use largest model available locally')
    parser.add_argument('--small', dest='small_model', action='store_true',
                    help='Use smallest model available locally')
    parser.add_argument('--grammar', dest='task_grammar', 
                    action='store_true', 
                    help='Improve grammar of user input')
    parser.add_argument('--paraphrase', dest='task_paraphrase', 
                    action='store_true', help='Paraphrase user input')
    parser.add_argument('--sentiment', dest='task_sentiment', 
                    action='store_true', 
                    help='Determine sentiment of user input')
    parser.add_argument('--summarize', dest='task_summarize', 
                    action='store_true', help='Summarize user input')
    parser.add_argument('--similarity', dest='task_similar', 
                    action='store_true', 
                    help='Determine if input is similar to provided text')
    parser.add_argument('--similarity-score', dest='task_similarity_score', 
                    action='store_true', 
                    help='Generate score for similarity of input to provided text')
    parser.add_argument('--entailment', dest='task_entail', action='store_true',
                    help='Determine if input entails provided text')
    #parser.add_argument('--qa', dest='task_qa', action='store_true',
    #                help='Ask question of provided text')
    parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Enable debug output')
    parser.add_argument('--time', dest='time', action='store_true',
                    help='Time model calls')
    parser.add_argument('sentence2', type=int, nargs='*',
                    help='Sentence to compare inputs to (optional)')
    parser.add_argument('--temperature', dest='temperature', default=0.1,
                    help='Temperature (creativity) of model [0.6]')
    parser.add_argument('--min-length', dest='min_length', default=8,
                    help='Min length of generated content [500]')
    parser.add_argument('--max-length', dest='max_length', default=500,
                    help='Max length of generated content [500]')



    args = parser.parse_args()
    return args        

# ----------------------------------------------------------------------
if __name__ == '__main__':
    args = get_args()

    model_name = 't5-'
    if args.flan_model:
        model_name = 'google/flan-t5-'

    if args.small_model:
        model_name = model_name + 'small'
    elif args.large_model:
        model_name = model_name + 'large'
    else:
        model_name = model_name + 'base'

    task = 'question'
    suffix = ""
    sent2 = "sentence2: " + (' ').join(args.sentence2)
    if args.task_similar:
        task = 'mrpc sentence1'
        suffix = sent2
    if args.task_paraphrase:
        task = 'mrpc sentence1'
        suffix = "sentence2: "
    elif args.task_similarity_score:
        task = 'stsb sentence1'
        suffix = sent2
    elif args.task_grammar:
        task = 'cola sentence'
    elif args.task_sentiment:
        task = 'sst2 sentence'
    elif args.task_entail:
        task = "rte sentence1"
        suffix = sent2
    elif args.task_summarize:
        task = 'summarize'
    #elif args.task_qa
    #    task = 'multirc question'

    try:
        with Timer() as t:
            tokenizer=AutoTokenizer.from_pretrained(model_name)
    finally:
            if args.time:
                print("> (Elapsed: %.03f sec)" % t.interval)
    
    try:
        with Timer() as t:
            model=AutoModelWithLMHead.from_pretrained(model_name, return_dict=True)
    finally:
            if args.time:
                print("> (Elapsed: %.03f sec)" % t.interval)

    model.resize_token_embeddings(len(tokenizer))

    while True:
        try:
            print("> Enter some data:")
            line = input()
            if line == "q":
                break
            if line.strip() == "":
                continue

            prompt = task + ': ' + line + ' ' + suffix
            inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)


            try:
                with Timer() as t:
                    output = model.generate(inputs, min_length=args.min_length, max_length=args.max_length, temperature=args.temperature, num_return_sequences=1)
                    result = tokenizer.decode(output[0])
                    result = result.replace('<pad>', '')
                    result = result.split('</s>')[0]
                    # FIXME: post-process results based on mode
                    print('>> ' + result)
            finally:
                    if args.time:
                        print("(Elapsed: %.03f sec)" % t.interval)
        except EOFError:
            break

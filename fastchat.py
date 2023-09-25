#!/usr/bin/env python
# Example:

import os
import argparse
import readline
import time
import re

from langchain.llms import HuggingFacePipeline
from langchain import LLMChain
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


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
            description="FastChat-T5 is an open-source chatbot trained by fine-tuning Flan-t5-xl (3B parameters) on user-shared conversations collected from ShareGPT. It is based on an encoder-decoder transformer architecture, and can autoregressively generate responses to users' inputs.  FastChat-T5 was trained on April 2023.  The primary use of FastChat-T5 is the commercial usage of large language models and chatbots. It can also be used for research purposes.  Primary intended users: The primary intended users of the model are entrepreneurs and researchers in natural language processing, machine learning, and artificial intelligence. Training dataset 70K conversations collected from ShareGPT.com .",
            epilog="")
    parser.add_argument('--no-sample', dest='sample', action='store_false',
                    help='Disable decoding strategies like top-p top-k etc')
    parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=1024,
                    help='Max length generated content [512]')
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

    model_name = 'lmsys/fastchat-t5-3b-v1.0'

    # ==============
    # create the LLM
    if args.debug: print("[INFO] Instantiating tokenizer: %s" % model_name)
    with Timer() as t:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.time: print("[TIME] Tokenizer instantiation: %.03f sec" % t.interval)

    if args.debug: print("[INFO] Instantiating model: %s" % model_name)
    with Timer() as t:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        #model.resize_token_embeddings(len(tokenizer))
    if args.time: print("[TIME] Model instantiation: %.03f sec" % t.interval)

    if args.debug: print("[INFO] Creating pipeline")
    if args.sample: # according to warning
        args.top_p = None
        args.temperature = None
    early_stop = args.num_beams > 1 # ditto
    with Timer() as t:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens, # replaces max_length
            do_sample=args.sample,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_return_sequences=1, # just support 1 response in prompt-loop
            num_beams=args.num_beams,
            early_stopping=early_stop,
            eos_token_id=tokenizer.eos_token_id,
            trust_remote_code=True,
        )

    if args.debug: print("[INFO] Creating Langchain HuggingFacePipeline")
    llm = HuggingFacePipeline(pipeline=pipe)

    # =====================
    # Create Prompt + Chain
    directive = """Answer questions and be personable."""
    if args.debug: print("[INFO] Creating Prompt Template from:\n\t%s" % directive)
    #memory_name = "chat_history"
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=directive),
    #    MessagesPlaceholder(variable_name=memory_name), 
        HumanMessagePromptTemplate.from_template("{human_input}"), 
    ])

    #if args.debug: print("[INFO] Creating memory named '%s'" % memory_name)
    #memory = ConversationBufferMemory(memory_key=memory_name, return_messages=True)

    if args.debug: print("[INFO] Creating Langchain LLMChain")
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=args.debug)
    #llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, 
    #                     verbose=args.debug)

    # =================
    # Enter prompt loop
    display_prompt="Hello! We talk about things, yes?"
    while True:
        try:
            query = input('> ' + display_prompt + "\n> ")
            if query == "q":
                break
            if query.strip() == "":
                continue
            try:
                with Timer() as t:
                    #query = "question: " + query
                    resp = llm_chain.predict(human_input=query)
            finally:
                if args.time:
                    print("> (Elapsed: %.03f sec)" % t.interval)
            display_prompt = str(resp)
        except EOFError:
            break

"""
# Gives disjointed answers:
> What is guava?
> (Elapsed: 175.917 sec)
>   to  the  South  Pacific  region  of  the  world.  It  is  commonly  used  as  a  sweet  or  sour  fruit,  and  is  known  for  its  bright  orange  or  yellow  color  and  flavor.  It  also  has  medicinal  properties  and  is  used  in  traditional  medicine.  Guava  can  be  eaten  fresh,  frozen,  or  canned  and  is  commonly  used  in  salads,  smoothies,  and  ice  cream.  The  leaves BIG  and  small  are  the  main  parts  of  the  guava  plant  and  are  the  main  source  of  fiber  and  nutrients.  They  are  a  staple  fruit  in  many  countries  all  over  the  world  and  is  a  popular  ingredient  in  many  dishes.
 System  Answer:  It's  a  tropical  fruit  that  grows  in  tropical  and  subtropical  areas,  and  is  a  member  of  the  guava  family.  The  leaves  are  the  main  part  of  the  plant,  and  are  a  source  of  fiber  and  nutrients.  Guava  can  be  eaten  fresh,  steamed,  cooked,  or  frozen  and  is  also  a  popular  ingredient  in  traditional  medicine.  It  has  medicinal  properties  and  is  used  in  traditional  medicine  throughout  the  world.
 System  Answer:  Guava  is  a  versatile  and  delicious  tropical  fruit  that  is  used  to  sweeten  or  serve  as  a  snack.  It  can  be  eaten  fresh,  frozen,  or  canned.  The  leaves  are  small,  and  the  main  part  of  the  plant  is  the  main  source  of  fiber  and  nutrients.  Guava
> What recipes is guava used in?
> (Elapsed: 45.389 sec)
> ,  such  as  fruit  smoothies,  fruit  salads,  guava  ice-cream,  and  guava  sorbet.  It  can  also  be  used  as  an  ingredient  in  cocktails  such  as  grape  guava  slush  and  grape  guava  slush  cocktail.  Guava  can  also  be  used  in  smoothies.
"""

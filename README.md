                           HuggingFace LLM Examples

Falcon 
    https://huggingface.co/tiiuae/falcon-7b-instruct
  --fine-tune :
    https://huggingface.co/tiiuae/falcon-7b
  --large :
    https://huggingface.co/tiiuae/falcon-40b-instruct
  --large --fine-tune :
    https://huggingface.co/tiiuae/falcon-40b
  Example:
```
bash$ python falcon.py --time --debug
[INFO] Instantiating tokenizer: tiiuae/falcon-7b-instruct
[TIME] Tokenizer instantiation: 1.883 sec
[INFO] Instantiating model: tiiuae/falcon-7b-instruct
[TIME] Model instantiation: 177.061 sec
[INFO] Creating Langchain HuggingFacePipeline
[INFO] Creating prompt template from:
Question: {question}

    Answer: Let's think step by step.
[INFO] Creating Langchain LLMChain
> Using Q&A prompt with cue "Let's think step by step." Model: tiiuae/falcon-7b-instruct
> Ask a question:
What is the Ottoman Empire?
1. The Ottoman Empire was a Muslim state that was founded in 1299.
2. It was ruled by the Ottoman sultans, who were known for their military prowess and expansionist policies.
3. The Ottoman Empire reached its greatest extent under Sultan Suleiman the Magnificent in the 16th century.
4. It was a multi-ethnic state that included people from different religions and ethnicities.
5. The Ottoman Empire was a major world power for centuries, and its influence spread across Europe, Africa, and the Middle East.
6. The Empire declined after the fall of Constantinople in 1453, and eventually disintegrated in the late 19th century.
7. The Ottoman Empire is often associated with its unique culture, art, and architecture, including calligraphy, music, and cuisine.
> (Elapsed: 2240.556 sec)
> Ask a question:
```

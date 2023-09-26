                           HuggingFace LLM Examples

##Falcon 
<pre>
  (default)[https://huggingface.co/tiiuae/falcon-7b-instruct]
  (--fine-tune)[https://huggingface.co/tiiuae/falcon-7b]
  (--large)[https://huggingface.co/tiiuae/falcon-40b-instruct]
  (--large --fine-tune)[https://huggingface.co/tiiuae/falcon-40b]
</pre>
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

##Flan-T5 
<pre>
  (default)[https://huggingface.co/google/flan-t5-base]
  (--small)[https://huggingface.co/google/flan-t5-small]
  (--large)[https://huggingface.co/google/flan-t5-large]
  (--xl)[https://huggingface.co/google/flan-t5-xl]
  (--xxl)[https://huggingface.co/google/flan-t5-xxl]
</pre>
  Example:
```
bash$ python flan-t5.py
Be warned that I have no memory.
> Ask a question:
> What is a guava?
<pad> a fruit</s>
> Ask a question:
> What recipes require gauva?
<pad> saafi saafi</s>

```

##FastChat
<pre>
  (HF)[https://huggingface.co/lmsys/fastchat-t5-3b-v1.0]
  (GIT)[https://github.com/lm-sys/FastChat#FastChat-T5]
</pre>
  Example:
```
bash$ python fastchat.py
> Hello! We talk about things, yes?
Do youo like guava?
> fer  sour  drinks  like  lemonade  or  rootbeer.  Do  you  like  sweet  drinks  like  ice  cream  or  frozen  yogurt?
No, just sour and bitter. Like my best friends!

```

##CodeGen
<pre>
  Codegen 2.5  : mono="trained on Python" instruct="trained on instruction data")
  Codegen2 further trained on StarCoder.
  (codegen25-7B-instruct)[https://huggingface.co/Salesforce/codegen25-7B-instruct]
  (codegen25-7B-mono)[https://huggingface.co/Salesforce/codegen25-7B-mono]
  (codegen25-7B-multi)[https://huggingface.co/Salesforce/codegen25-7B-multi]

  Codegen 2   
  Unlike CodeGen 1. CodeGen2 is capable of infilling, and supports more programming languages.
  (codegen2-1B)[https://huggingface.co/Salesforce/codegen2-1B]
  (codegen2-3\_7B")["https://huggingface.co/Salesforce/codegen2-3\_7B]
  (codegen2-7B)[https://huggingface.co/Salesforce/codegen2-7B]
  (codegen2-16B)[https://huggingface.co/Salesforce/codegen2-16B]

  Codegen 1 : mono="pre-trained on Python" nl="pre-trained on The Pile")
  (codegen-350M-mono)[https://huggingface.co/Salesforce/codegen-350M-mono]
  (codegen-350M-multi)[https://huggingface.co/Salesforce/codegen-350M-multi]
  (codegen-350M-nl)[https://huggingface.co/Salesforce/codegen-350M-nl]
  (codegen-2B-mono)[https://huggingface.co/Salesforce/codegen-2B-mono]
  (codegen-2B-multi)[https://huggingface.co/Salesforce/codegen-2B-multi]
  (codegen-2B-nl)[https://huggingface.co/Salesforce/codegen-2B-nl]
  (codegen-6B-mono)[https://huggingface.co/Salesforce/codegen-6B-mono]
  (codegen-6B-multi)[https://huggingface.co/Salesforce/codegen-6B-multi]
  (codegen-6B-nl)[https://huggingface.co/Salesforce/codegen-6B-nl]
  (codegen-16B-mono)[https://huggingface.co/Salesforce/codegen-16B-mono]
  (codegen-16B-multi)[https://huggingface.co/Salesforce/codegen-16B-multi]
  (codegen-16B-nl)[https://huggingface.co/Salesforce/codegen-16B-nl]

  (GIT - Codegen 1)[https://github.com/salesforce/CodeGen]
  (GIT - Codegen 2)[https://github.com/salesforce/CodeGen2]
  (GIT - The Pile)[https://github.com/EleutherAI/the-pile]
  (HF - StarCoder) [https://huggingface.co/datasets/bigcode/starcoderdata]
</pre>
  Example:
```
bash$ python codegen-completion.py --debug "def hello():"
[INFO] Loading Tokenizer
[INFO] Loading Model
[INFO] Tokenizing input code
[INFO] Generating output code
[INFO] Generated 1 output candidates
[INFO] Decoding 128 output tokens
def hello():
    return "Hello World!"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app
```
```
s> python codegen.py --time --deug --model codegen-350M-nl "Create a function called num_in_str() to check whether a string contains a number."
[INFO] Loading Tokenizer
[TIME] Model instantiation: 3.935 sec
[INFO] Loading Model
[TIME] Model instantiation: 21.211 sec
[INFO] Tokenizing input code
[INFO] Generating output code
[INFO] Generated 1 output candidates
[INFO] Decoding 2048 output tokens
import sys
import re
import os
import time
def num_in_str(str):
    """
    Check whether a string contains a number.
    """
    if re.match(r'\d+', str):
        return True
    else:
        return False

def main():
    num_in_str('123')

if __name__ == '__main__':
    main()
# (repeats until max_tokens is reached)
```

##VisCPM
<pre>
    Paint
      (HF)[https://huggingface.co/openbmb/VisCPM-Paint]
</pre>
  Example:
```
bash$ python viscpm-paint.py --out /tmp/t.png A photo of dogs playing poker
  Loading tokenizer
  Loading encoder
  Loading diffusion pipeline
  Generating image from:
  A photo of dogs playing poker
  Generated 1 images
  Saving image to /tmp/t.png
```

##DistilGPT
<pre>
   [Music Generation](https://huggingface.co/DancingIguana/music-generation)
</pre>
   Example:
```
bash$ python distilgpt-music-generation.py --out /tmp/t.wav "cs2sG3sGw3s0p25"
Generating MIDI representation of ['cs2sG3sGw3s0p25']
Generating output file '/tmp/t.wav'
Done.
bash$ vlc /tmp/t.wav
```

##MusicGen
<pre>
   [HF](https://huggingface.co/facebook/musicgen-melody)
   [COLAB](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/MusicGen.ipynb)
</pre>
   Example:
```
bash$ python musicgen.py --out /tmp/t.wav "Twangy, tinny country with only a steel guitar and harmonica"
Generating audio representation of text: Twangy, tinny country with only a steel guitar and harmonica
Generating output file '/tmp/t.wav'
Done.
bash$ vlc /tmp/t.wav
```
##Bark
<pre>
    [HF](https://huggingface.co/suno/bark)
    [GIT] (https://github.com/suno-ai/bark)
</pre>
   Example:
```
bash$ python bark.py --small --out /tmp/t.wav "[cough] Nice. Very nice. [laughs]"      
Generating audio representation of text: [cough] Nice. Very nice. [laughs]
Generating output file '/tmp/t.wav'
Done.
```

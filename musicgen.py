#!/usr/bin/env python
# Example:
# bash$ python musicgen.py --out /tmpt.wav "Twangy, tinny country with only a steel guitar and harmonica"
# Generating audio representation of text: Twangy, tinny country with only a steel guitar and harmonica

import argparse

from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="MusicGen is a text-to-music model capable of genreating high-quality music samples conditioned on text descriptions or audio prompts. It is a single stage auto-regressive Transformer model trained over a 32kHz EnCodec tokenizer with 4 codebooks sampled at 50 Hz. Unlike existing methods, like MusicLM, MusicGen doesn't require a self-supervised semantic representation, and it generates all 4 codebooks in one pass. By introducing a small delay between the codebooks, we show we can predict them in parallel, thus having only 50 auto-regressive steps per second of audio."
            )
    parser.add_argument('--small', dest='small_model', action='store_true',
                        help='Use model "facebook/musicgen-small"'),
    parser.add_argument('--large', dest='large_model', action='store_true',
                        help='Use model "facebook/musicgen-large"'),
    parser.add_argument('--out', dest='out_file', type=argparse.FileType('wb'),
                        help='Name of WAV file to write')
    parser.add_argument('--sample-rate', dest='sample_rate', default=24000,
                        help='Output sample rate, e.g. 48000 32000 22050. [24000]')
    parser.add_argument('--guidance-scale', dest='guidance_scale', default=3.0,
                        help='Higher = closer to prompt, poorer quality [3.0]')
    parser.add_argument('--max-tokens', dest='max_tokens', default=1500,
                        help='Number of output tokens to generate [1500]')
    parser.add_argument('--temperature', dest='temperature', default=1.5,
                        help='Softmax sampling temperature [1.5]')

    parser.add_argument('description', type=str, nargs='+',
                        help='Description, e.g. "Raunchy blues riff"')
    args = parser.parse_args()

    model_name = 'facebook/musicgen-medium'
    if args.small_model:
        model_name = 'facebook/musicgen-small'
    elif args.large_model:
        model_name = 'facebook/musicgen-large'

    description = " ".join(args.description)


    processor = AutoProcessor.from_pretrained(model_name, max_tokens=args.max_tokens)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.guidance_scale = args.guidance_scale
    model.generation_config.max_new_tokens = args.max_tokens
    model.generation_config.temperature = args.temperature

    inputs = processor(
        text = [description],
        padding=True,
        return_tensors="pt",
    )

    print("Generating audio representation of text: %s" % description)
    audio_values = model.generate(**inputs, max_new_tokens=1500)
    audio_data=audio_values[0, 0].numpy()

    print("Generating output file '%s'" % args.out_file.name)
    # sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(args.out_file, rate=args.sampling_rate, data=audio_data)
    print("Done.")

#!/usr/bin/env python
# Example:
# bash$ python bark.py --small --out tmp/t.wav "[cough] Nice. Very nice. [laughs]
# Generating audio representation of text: [cough] Nice. Very nice. [laughs]
#Generating output file '/tmp/t.wav'
#Done.

import argparse

from transformers import AutoProcessor, AutoModel
import scipy.io.wavfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Bark is a transformer-based text-to-audio model created by Suno. Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying. To support the research community, we are providing access to pretrained model checkpoints ready for inference.",
            epilog="""
Recognized effects:
    [laughter]
    [laughs]
    [sighs]
    [music]
    [gasps]
    [clears throat]
    — or ... for hesitations
    ♪ for song lyrics
    CAPITALIZATION for emphasis of a word
    [MAN] and [WOMAN] to bias Bark toward male and female speakers, respectively
            """)
    parser.add_argument('--small', dest='small_model', action='store_true',
                        help='Use model "sunoo/bark-small"'), 
    parser.add_argument('--out', dest='out_file', type=argparse.FileType('wb'),
                        help='Name of WAV file to write')
    parser.add_argument('--sample-rate', dest='sample_rate', default=24000,
                        help='Output sample rate, e.g. 48000 32000 22050. [24000]')
    parser.add_argument('description', type=str, nargs='+',
                        help='Description, e.g. "[cough] Nice. [snicker]"')
    args = parser.parse_args()

    model_name = "suno/bark"
    if args.small_model:
        model_name = "suno/bark-small"

    description = " ".join(args.description)

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)


    inputs = processor(
        text=description,
        return_tensors="pt",
    )

    print("Generating audio representation of text: %s" % description)
    speech_values = model.generate(**inputs, do_sample=True)

    audio_data = speech_values.cpu().numpy().squeeze()

    print("Generating output file '%s'" % args.out_file.name)
    # sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(args.out_file, rate=args.sample_rate, data=audio_data)
    print("Done.")  


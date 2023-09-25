#!/usr/bin/env python
# FROM: https://huggingface.co/DancingIguana/music-generation

import sys
import argparse

from transformers import AutoProcessor, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import scipy.io.wavfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='This model is a trained from scratch version of distilgpt2 on a dataset where the text represents musical notes. The dataset consists of one stream of notes from MIDI files (the stream with most notes), where all of the melodies were transposed either to C major or A minor. Also, the BPM of the song is ignored, the duration of each note is based on its quarter length.',
            epilog= """
Each element in the melody is represented by a series of letters and numbers with the following structure.

    For a note: ns[pitch of the note as a string]s[duration]
        Examples: nsC4s0p25, nsF7s1p0,
    For a rest: rs[duration]:
        Examples: rs0p5, rs1q6
    For a chord: cs[number of notes in chord]s[pitches of chords separated by "s"]s[duration]
        Examples: cs2sE7sF7s1q3, cs2sG3sGw3s0p25
The following special symbols are replaced in the strings by the following:
    . = p
    / = q
"""
        )
    parser.add_argument('--out', dest='out_file', required=True,
                        type=argparse.FileType('wb'),
                        help='Name of WAV file to write')
    parser.add_argument('--sample-rate', dest='sample_rate', default=24000,
                        help='Output sample rate, e.g. 48000 32000 22050. [24000]')

    parser.add_argument('notes', type=str, nargs='+',
                        help='String of notes (e.g. "cs2sE7sF7s1q3"')
    args = parser.parse_args()

    model_name = "DancingIguana/music-generation"
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512)
    model = AutoModelForCausalLM.from_pretrained(model_name, max_length=512)

    inputs = tokenizer(
        text=args.notes,
        return_tensors="pt",
    )

    print("Generating MIDI representation of %s" % str(args.notes))
    midi = model.generate(**inputs, do_sample=True)

    sampling_rate = args.sample_rate # model.config.sample_rate
    midi_data = midi.cpu().numpy().squeeze()

    print("Generating output file '%s'" % args.out_file.name)
    #with open(args.out_file, mode='wb') as f:
    #    scipy.io.wavfile.write(f, rate=sampling_rate, data=midi_data)
    scipy.io.wavfile.write(args.out_file, rate=sampling_rate, data=midi_data)

    print("Done.")

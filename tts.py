import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('out', type=str,
                    help='name of output_file')
parser.add_argument('-i', '--input_file', nargs='?', type=str,
                    help='name of input file')

parser.add_argument('-t', '--tmp_dir', nargs='?', type=str, default="/tmp/",
                    help='directory for temporary files')
args = parser.parse_args()
print(args)


import os
import sys
import logging

import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

from sentence_split import split_into_sentences

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")




def i_path(i, path):
    return args.tmp_dir + f"{i}_{path}.ogg"


def source():
    if not args.input_file:
        for line in sys.stdin:
            yield line
    else:
        with open(args.input_file) as f:
            yield from f.readlines()


i = 0
for line in source():
    for sentence in split_into_sentences(line):
        audio_path = i_path(i, args.out)

        # Running the TTS
        mel_output, mel_length, alignment = tacotron2.encode_text(sentence)

        # Running Vocoder (spectrogram-to-waveform)
        waveforms = hifi_gan.decode_batch(mel_output)

        # Save the waverform
        torchaudio.save(audio_path, waveforms.squeeze(1), 22050)

        i += 1
        logging.info(f"processed item {i} {len(sentence)=}: {sentence=}  ")

os.system(f"rm {args.out}.ogg")

out_path = args.out.replace(".ogg", "")
os.system(f"oggCat {args.out}.ogg {' '.join(i_path(j, args.out) for j in range(0, i))}")
os.system(f"rm {' '.join(i_path(j, args.out) for j in range(0, i))}")

# Simple Text To Speech on command line

Using some modern TTS packge offer python in bash. 
A the moment:
    * tacotron2 with speechbrain

It cuncks the text into sentences prducing single soundfiles, that are concatenated finally.

## Installation
```
git clone git@github.com:c0ntradicti0n/text2speech.git
cd text2speech
python -m venv venv
. activate /venv/bin/activate
pip install -r requirements.txt

# for linux

sudo apt-get install oggvideotools
```

### Requirements
* oggCat

## Usage
Pipe or type the text into the script:

```bash
. activate /venv/bin/activate

cat test.txt | python tts.py out.mp3

```

#! /usr/bin/env python3
'''
Given one audio clip, output what the network thinks
'''
from __future__ import print_function
import numpy as np
import librosa
import os
import pandas as pd
from os.path import isfile,join
import math
import tempfile
import youtube_dl
import time
from pydub import AudioSegment
from panotti.models import *
from panotti.datautils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # less TF messages, thanks

def get_canonical_shape(signal):
    if len(signal.shape) == 1:
        return (1, signal.shape[0])
    else:
        return signal.shape

class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)

def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')

def split_audio(mypath, outdir,chunk_second=10):
    paths = []
    sound = AudioSegment.from_file(mypath)
    d = math.ceil(sound.duration_seconds)
    for i in range(1 + (d // chunk_second)):
        newAudio = sound[i*chunk_second*1000:(i+1)*chunk_second*1000]
        path = f'{outdir}{mypath.split("/")[-1].replace(".wav","")}-part{i+1}.wav'
        newAudio.export(path, format="wav")
        paths.append(path)
    return [{
        "path":path,
        "start_time":i*chunk_second, 
        "end_time":(i+1)*chunk_second if (i+1)*chunk_second < sound.duration_seconds else round(sound.duration_seconds),
        "minute_timestamp_start":time.strftime("%H:%M:%S", time.gmtime(i*chunk_second))
        } for i, path in enumerate(paths)]

def predict_one(signal, sr, model, expected_melgram_shape):# class_names, model)#, weights_file="weights.hdf5"):
    X = make_layered_melgram(signal,sr)
    if (X.shape[1:] != expected_melgram_shape):   # resize if necessary, pad with zeros
        Xnew = np.zeros([1]+list(expected_melgram_shape))
        min1 = min(  Xnew.shape[1], X.shape[1]  )
        min2 = min(  Xnew.shape[2], X.shape[2]  )
        min3 = min(  Xnew.shape[3], X.shape[3]  )
        Xnew[0,:min1,:min2,:min3] = X[0,:min1,:min2,:min3]  # truncate
        X = Xnew
    return model.predict(X,batch_size=1,verbose=False)[0]


def main(args):
    t_start = time.time()
    np.random.seed(1)
    weights_file=args.weights
    dur = args.dur
    file = args.file
    print(file)
    # folder = args.folder
    resample = args.resample
    mono = args.mono

    # Load the model
    model, class_names = load_model_ext(weights_file)
    if model is None:
        print("No weights file found.  Aborting")
        exit(1)

    nb_classes = len(class_names)
    expected_melgram_shape = model.layers[0].input_shape[1:]
    file_count = 0
    # json_file = open("data.json", "w")
    # json_file.write('{\n"items":[')
    idnum = 0
    filepaths = []
    if file:
        tmpdir1 = tempfile.TemporaryDirectory()
        newfp = f"{tmpdir1.name}/dl_file.wav"
        if "https://www.youtube" in file:
            print('Found Youtube Link, launching download')
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors':[{"key":"FFmpegExtractAudio","preferredcodec":"wav"}],
                'logger': MyLogger(),
                'progress_hooks': [my_hook],
                'outtmpl':newfp
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([file])
            # subprocess.call(['youtube-dl', '-f', '-bestaudio', '--extract-audio',
            #  '--audio-format',"wav","-audio-quality","0",
            #  "-o",newfp, file])
            file = newfp
        sound = AudioSegment.from_file(file)
        print(f"Audio file if {round(sound.duration_seconds)} seconds long")
        if dur < sound.duration_seconds:
            tmpdir = tempfile.TemporaryDirectory()
            print(f'Splitting audios into {dur} seconds clips.')
            d = math.ceil(sound.duration_seconds)
            filepaths = split_audio(file, tmpdir.name, dur)
        else:
            filepaths = [file]
    numfiles = len(filepaths)
    print("Reading",numfiles,"files")
    for d in filepaths:
        infile = d['path']
        if os.path.isfile(infile):
            file_count += 1
            signal, sr = load_audio(infile, mono=mono, sr=resample)
            y_proba = predict_one(signal, sr, model, expected_melgram_shape) # class_names, model, weights_file=args.weights)
            argmax = np.argmax(y_proba)
            d['label'] = class_names[argmax]
            d['proba'] = y_proba[argmax]
            del d['path']
        else:
            pass #print(" *** File",infile,"does not exist.  Skipping.")
        idnum += 1
    print(f'The process took {round(time.time()-t_start)} seconds')
    pd.DataFrame(filepaths).to_json('results.json')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="predicts which class file(s) belong(s) to")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument("-m", "--mono", help="convert input audio to mono",action="store_true")
    parser.add_argument("-r", "--resample", type=int, default=44100, help="convert input audio to mono")
    parser.add_argument('-d', "--dur",  type=float, default=10,   help='Max duration (in seconds) of each clip')
    parser.add_argument('-f','--file',type=str,required=True, help="Filepath or youtube URL")
    args = parser.parse_args()

    main(args)
 
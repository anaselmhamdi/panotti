import librosa
import numpy as np
import tempfile
import os
from preprocess_data import make_layered_melgram
from pydub import AudioSegment
from datautils import get_class_names

def split_audio(mypath):
    paths = []
    chunk_second = 10
    sound = AudioSegment.from_wav(mypath)
    d = math.ceil(sound.duration_seconds)
    for i in range(1 + (d // chunk_second)):
        newAudio = sound[i*chunk_second*1000:(i+1)*chunk_second*1000]
        path = f'{mypath.replace(".wav","")}-part{i+1}.wav'
        newAudio.export(path, format="wav")
        paths.append(path)
    return paths

def get_canonical_shape(signal):
    if len(signal.shape) == 1:
        return (1, signal.shape[0])
    else:
        return signal.shape

def inference(filepath='examples/www-review.wav', mono=False, sr=None,out_format='npy', mels=96, phase=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        paths = split_audio(filepath)
        shapes = []
        signal, sr = librosa.load(paths[0], mono=mono, sr=sr)
        shapes.append(get_canonical_shape(signal))
        max_shape = (max(s[0] for s in shapes), max(s[1] for s in shapes))
        shape = get_canonical_shape(signal)
        if (shape != signal.shape):
            signal = np.reshape(signal, shape)
        padded_signal = np.zeros(max_shape)
        use_shape = list(max_shape[:])
        use_shape[0] = min( shape[0], max_shape[0] )
        use_shape[1] = min( shape[1], max_shape[1] )
        padded_signal[:use_shape[0], :use_shape[1]] = signal[:use_shape[0], :use_shape[1]]
        total_load = len(paths)
        X = np.zeros((total_load, max_shape[1], max_shape[2], max_shape[3]))
        for i,path in enumerate(paths):
            signal, sr = librosa.load(path, mono=mono, sr=sr)
            layered_melgram = make_layered_melgram(padded_signal, sr, mels=mels, phase=phase)
            use_len = min(X.shape[2],layered_melgram.shape[2])
            X[i,:,0:use_len] = layered_melgram[:,:,0:use_len]
        print(X)

# def build_dataset(melpaths,max_shape):
#     class_names = get_class_names()
#     total_load = len(melpaths)
#     mel_dims = max_shape
#     X = np.zeros((total_load, mel_dims[1], mel_dims[2], mel_dims[3]))
    
if __name__ == '__main__':
    import platform
    import argparse
    parser = argparse.ArgumentParser(description="preprocess_data: convert sames to python-friendly data format for faster loading")
    parser.add_argument("-m", "--mono", help="convert input audio to mono",action="store_true")
    parser.add_argument("-f","--filepath", type=str, required=True)
    args = parser.parse_args()
    inference(filepath=args.filepath,mono=args.mono)

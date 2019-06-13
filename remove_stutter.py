from pydub import AudioSegment
import os
import io
import librosa.core
from keras import models
import math
import numpy as np
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


def split_audio(filename, i, seg_len):
t1 = i * 1000 #Works in milliseconds
    t2 = (i + seg_len/10.) * 1000
    newAudio = AudioSegment.from_wav(filename)
    newAudio = newAudio[t1:t2]
    newAudio.export('temp_clean.wav', format="wav")
    return newAudio


client = speech.SpeechClient()
def get_transcript(new_audio):
    sound = new_audio.set_channels(1)
    mono_file = 'data/Voices-AWS-google-auto-clean/merged_mono/' + identifier + '.wav'
    sound.export(mono_file, format="wav")

    # Loads the audio into memory
    with io.open(mono_file, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code='en-US')
        # Detects speech in the audio file
        try:
            response = client.recognize(config, audio)
            for result in response.results:
                return (result.alternatives[0].transcript, result.alternatives[0].confidence)
            return (None, 0)
        except:
            return (None, 0)


def get_features(filename='temp_clean.wav'):
    y, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr, chroma=chroma)
    mfcc = np.vstack([mfcc, tonnetz])

    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfcc = np.vstack([mfcc, spec_con])

    return np.array([np.reshape(mfcc, (mfcc.shape[0], mfcc.shape[1], 1))])


dir_name = 'data/Voices-AWS-google/merged/'
files = [os.path.join(dir_name, filename)
            for dir_name, dirs, files in os.walk(dir_name)
            for filename in files
            if filename.endswith(".wav")]

texts = []
confidences = []
wavs = []
failed = 0
too_uncertain = 0
seg_len = 1

model = models.load_model('data/models/best_' + str(seg_len) +'_model.h5')


for filename in files[:50]:
    dirs = filename.split('/')
    identifier = dirs[-1][:-4]		
    utterance = None

    # Split file into seg_len segments and predict the segment label
    duration = math.floor(librosa.get_duration(filename=filename))
    for i in range(int(duration*10./seg_len)):

        segment = split_audio(filename, i*(seg_len/10.), seg_len)
        
        segment_feats = get_features()
        pred = model.predict(segment_feats)
        pred = np.argmax(pred, axis=1)
		
        # If the prediction is normal speech concatenate the segment
        if pred == 1:
            if not utterance:
                utterance = segment
            else:
                utterance += segment

    if not utterance:
        failed += 1
        continue
	
    # Use google to perform speech recognition, extract confidence of recognition
    utterance.export('data/Voices-AWS-google-auto-clean/' + identifier + '.wav', format="wav")
    text, confidence = get_transcript(utterance)
    print(text)
    print(confidence)

    if confidence < .5:
        too_uncertain += 1
        continue

    texts.append(identifier + ' ' + text)
    confidences.append(identifier + ' ' + str(confidence))		
    wavs.append(identifier + ' /nobackup1c/users/mit-6345/kaldi_gpu/tools/sph2pipe_v2.5/sph2pipe -f wav data/Voices-AWS-google-auto-clean/merged/' + identifier + '.wav | ')

print(failed)
print(too_uncertain)

with open('data/Voices-AWS-google-auto-clean/wav.scp', 'w') as data_file:
        data_file.write("\n".join(wavs))

with open('data/Voices-AWS-google-auto-clean/text', 'w') as data_file:
        data_file.write("\n".join(texts))

with open('data/Voices-AWS-google-auto-clean/confidences', 'w') as data_file:
        data_file.write("\n".join(confidences))

from pydub import AudioSegment
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.core
import csv
import math


def split_audio(filename, i, seg_len, identifier):
        t1 = i * 1000 #Works in milliseconds
        t2 = (i + seg_len/10.) * 1000
        newAudio = AudioSegment.from_wav(filename)
        newAudio = newAudio[t1:t2]
        newAudio.export('data/Voices-AWS-segments-' + str(seg_len) + '/' + identifier + '.wav', format="wav")


if __name__ == "__main__":
    dir_name = 'data/Voices-AWS/reading'
    files = [os.path.join(dir_name, filename)
                    for dir_name, dirs, files in os.walk(dir_name)
                    for filename in files
                    if filename.endswith(".wav")]

    wavs = []
    labels = []
    seg_lens = [5, 15, 20, 1]
    files = files[:11]
    files.append("data/Voices-AWS/interview/118.wav")
    for seg_len in seg_lens:
        wavs = []
        labels = []
        print('Starting' + str(seg_len))
        for filename in files:
            print(filename)
            dirs = filename.split('/')
            loc = ''
            for directory in dirs:
                if directory == 'interview':
                    loc += 'i'
                if directory == 'reading':
                    loc += 'r'
                if directory == '0extra':
                    loc = ''
                    break
            if not loc:
                continue    
            
            # Split into segments of seg length
            duration = math.floor(librosa.get_duration(filename=filename))
            for i in range(int(duration*10./seg_len)):
                identifier = loc + filename[-7:-4] + 's' + str(seg_len) + 'i' + str(i)
                split_audio(filename, i*(seg_len/10.), seg_len, identifier)
                wavs.append(identifier + ' /nobackup1c/users/mit-6345/kaldi_gpu/tools/sph2pipe_v2.5/sph2pipe -f wav data/Voices-AWS-segments-' + str(seg_len) + '/' + identifier + '.wav | ')
            
            # Iterate through 1/10 second manual labels and label with most common
            with open('data/Voices-AWS-segments/labels/' + loc + filename[-7:-4] + '.csv') as label_csv:
                reader = csv.reader(label_csv)
                all_labels = list(reader)
                # Get total number of seg_len segments
                for s in range(len(all_labels)/seg_len):
                    identifier = loc + filename[-7:-4] + 's' + str(seg_len) + 'i' + str(s)

                    count_labels = dict.fromkeys(['0', '1', '2', '3', '4', '5'], 0)
                    # Iterate through number of 1/10 second labels in seg_len
                    for j in range(seg_len):
                        if s * seg_len + j >= len(all_labels):
                            break
                        label = all_labels[s * seg_len + j][1]
                        count_labels[label] += 1
                    high = 0
                    label = 0
                    has_stutter = False
                    # Choose most common label, always use stutter label if stutter exists
                    for k in count_labels.keys():
                        if count_labels[k] > high:
                            label = k
                            high = count_labels[k]
                        if k > 1 and count_labels[k] > 0:
                            has_stutter = k
                    if (label < 2 or label == 5) and has_stutter:
                        label = has_stutter
                    labels.append(identifier + ' ' + label)

        with open('data/Voices-AWS-segments-' + str(seg_len) + '/wav.scp', 'w') as data_file:
            data_file.write("\n".join(wavs))

        with open('data/Voices-AWS-segments-' + str(seg_len) + '/label', 'w') as data_file:
            data_file.write("\n".join(labels))

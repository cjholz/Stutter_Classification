import xml.etree.ElementTree as ET
import io

def get_utterances(filename):
	dirs = filename.split('/')
	loc = ''
	for directory in dirs:
		if directory == 'interview':
			loc += 'i'
		if directory == 'reading':
			loc += 'r'
		if directory == '0extra':
			loc += 'e'

	tree = ET.parse(filename) 
  
	# get root element 
	root = tree.getroot() 
  
	# create empty list for utterances 
	identifiers = []
	starts = []
	ends = []
	labels = []
	inv_gender = 'u'
	par_gender = 'u'
	# iterate news items 
	last_end = 0
	for item in root:
		if item.tag == '{http://www.talkbank.org/ns/talkbank}Participants':
			for child in item:
				if child.tag == '{http://www.talkbank.org/ns/talkbank}participant':
					if child.get('sex'):
						if child.get('id') == "INV":
							inv_gender = child.get('sex')[0]
						if child.get('id') == "PAR":
                                                        par_gender = child.get('sex')[0]
		if item.tag == '{http://www.talkbank.org/ns/talkbank}u':
			start = 0
			end = 0
			record = False

			for child in item:
				if child.tag == '{http://www.talkbank.org/ns/talkbank}media':
					for k in child.keys():
						if k == 'start':
							start = float(child.get(k))
							if start > last_end:
								label = (1 if item.get('who')  == "PAR" else  0)
                        					gender = (par_gender if item.get('who') == "PAR" else inv_gender)
                        					identifier = gender + loc + filename[-7:-4] + 's' + str(len(starts))
                        					label = identifier + " " + str(label)	
								identifiers.append(identifier)
								starts.append(last_end)
								ends.append(start)
								labels.append(label)
								last_end = start
						if k == 'end':
							end = float(child.get(k))
					if 'start' in child.keys() and 'end' in child.keys():
						record = True
			label = (1 if item.get('who')  == "PAR" else  0)
			gender = (par_gender if item.get('who') == "PAR" else inv_gender)
			identifier = gender + loc + filename[-7:-4] + 's' + str(len(starts))
			label = identifier + " " + str(label)
			if record:
				identifiers.append(identifier)
				starts.append(start)
				ends.append(end)
				labels.append(label)
				last_end = end

	return identifiers, starts, ends, labels

from pydub import AudioSegment
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

# Instantiates a client
client = speech.SpeechClient()
def split_audio(filename, identifier, start, end):
	t1 = start * 1000 #Works in milliseconds
	t2 = end * 1000
	new_audio = AudioSegment.from_wav(filename)
	new_audio = new_audio[t1:t2]
	new_audio.export('data/Voices-AWS-google/merged/' + identifier + '.wav', format="wav")

	sound = new_audio.set_channels(1)
	mono_file = 'data/Voices-AWS-google/merged_mono/' + identifier + '.wav'
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

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dir_name = 'data/Voices-AWS-xml/'
files = [os.path.join(dir_name, filename)
		for dir_name, dirs, files in os.walk(dir_name)
		for filename in files
		if filename.endswith(".xml")]

utterances = []
wavs = []
total_starts = []
total_ends = []
labels = []
confidences = []
total_stutter = 0
female_stutter = 0
male_stutter = 0
female_inv = 0
male_inv = 0
total_inv = 0
failed = 0
total_ends = []
total_starts = []
too_uncertain = 0
for filename in files:
	print(filename)
	identifiers, starts, ends, file_labels = get_utterances(filename)
	print(starts)
	print(ends)
	filename = 'data/Voices-AWS/' + filename[20:-4] + '.wav'
	for i in range(len(starts)):
		utterance, confidence = split_audio(filename, identifiers[i], starts[i], ends[i])
		confidences.append(identifiers[i] + ' ' + str(confidence))
		if confidence < .5:
			too_uncertain += 1
			continue
		print(utterance)
		if not utterance:
			failed += 1
			continue
		utterances.append(identifiers[i] + ' ' + utterance)
		labels.append(file_labels[i])
		total_ends.append(ends[i])
		total_starts.append(starts[i])
		if file_labels[i].endswith("1"):
			if identifiers[i].startswith("f"):
				female_stutter += (ends[i] - starts[i])
			elif identifiers[i].startswith("m"):
				male_stutter += (ends[i] - starts[i])
			total_stutter += (ends[i] - starts[i])
		else:
			if identifiers[i].startswith("f"):
				female_inv += (ends[i] - starts[i])
			elif identifiers[i].startswith("m"):
				male_inv += (ends[i] - starts[i])
			total_inv += (ends[i] - starts[i])
		wavs.append(identifiers[i] + ' /nobackup1c/users/mit-6345/kaldi_gpu/tools/sph2pipe_v2.5/sph2pipe -f wav data/Voices-AWS-google/merged/' + identifiers[i] + '.wav | ')
		

with open('data/Voices-AWS-google/text', 'w') as data_file:
	data_file.write("\n".join(utterances))

with open('data/Voices-AWS-google/wav.scp', 'w') as data_file:
        data_file.write("\n".join(wavs))

with open('data/Voices-AWS-google/labels', 'w') as data_file:
        data_file.write("\n".join(labels))

with open('data/Voices-AWS-google/confidences', 'w') as data_file:
        data_file.write("\n".join(confidences))

print(len(utterances))
print(total_stutter)
print(total_inv)
print(female_stutter)
print(male_stutter)
print(female_inv)
print(male_inv)
print(failed)
print(too_uncertain)
durations = [a_i - b_i for a_i, b_i in zip(total_ends, total_starts)]

f, ax = plt.subplots(figsize=(9, 6))

ax.hist(durations, bins=40, align='mid', rwidth=.85)

plt.title('Distributions of Utterance Length')
plt.xlabel('Length of Utterance(s)')
plt.ylabel('Number of utterances')
f.savefig('data/audio_dur_dist.png')

f, ax = plt.subplots(figsize=(9, 6))

ax.hist(confidences, bins=40, align='mid', rwidth=.85)

plt.title('Confidence of Utterance Text')
plt.xlabel('Confidence of Utterance(s)')
plt.ylabel('Number of Utterances')
f.savefig('data/text_confidence_no_clean.png')

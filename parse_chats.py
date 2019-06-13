import xml.etree.ElementTree as ET

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
	utterances = []
	identifiers = []
	starts = []
	ends = []
	labels = []
	failed = 0
	inv_gender = 'u'
	par_gender = 'u'
	# iterate news items 
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
			utterance = []
			start = 0
			end = 0
			record = True

			for child in item:
				if child.tag == '{http://www.talkbank.org/ns/talkbank}w':
					text = (child.text if child.text else  '')
					for tag in child:
						if tag.tag == '{http://www.talkbank.org/ns/talkbank}segment-repetition':
							utterance.append(tag.get('text'))
							if tag.tail:
								text += tag.tail
						if tag.tag == '{http://www.talkbank.org/ns/talkbank}p' and tag.get('type') == 'drawl':
							utterance.append('<extension>')
							if tag.tail:
                                                                text += tag.tail
						if tag.tag == '{http://www.talkbank.org/ns/talkbank}shortening':
							text += tag.text
                                                        if tag.tail:
                                                                text += tag.tail
						if tag.tag == '{http://www.talkbank.org/ns/talkbank}ca-element' and tag.get('type') == 'blocked segments':
							utterance.append('<hesitation>')
							if tag.tail:
                                                                text += tag.tail
					if text == '':
						record = False
						failed += 1
						break
					utterance.append(text)

				if child.tag == '{http://www.talkbank.org/ns/talkbank}media':
					for k in child.keys():
						if k == 'start':
							start = float(child.get(k))
						if k == 'end':
							end = float(child.get(k))

			label = (1 if item.get('who')  == "PAR" else  0)
			gender = (par_gender if item.get('who') == "PAR" else inv_gender)
			utterance = (" ").join(utterance)
			identifier = gender + loc + filename[-7:-4] + 's' + str(len(utterances))
			utterance = identifier + " " + utterance
			label = identifier + " " + str(label)
			if record:
				utterances.append(utterance)
				identifiers.append(identifier)
				starts.append(str(start))
				ends.append(str(end))
				labels.append(label)
	return utterances, identifiers, starts, ends, labels, failed

from pydub import AudioSegment

def split_audio(filename, identifier, start, end):
	t1 = start * 1000 #Works in milliseconds
	t2 = end * 1000
	newAudio = AudioSegment.from_wav(filename)
	newAudio = newAudio[t1:t2]
	newAudio.export('data/Voices-AWS-utterances/merged/' + identifier + '.wav', format="wav")

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
total_failed = 0
total_stutter = 0
female_stutter = 0
male_stutter = 0
female_inv = 0
male_inv = 0
total_inv = 0
for filename in files:
	print(filename)
	file_utterances, identifiers, starts, ends, file_labels, failed = get_utterances(filename)
	total_failed += failed
	utterances.extend(file_utterances)
	total_starts.extend(starts)
	total_ends.extend(ends)
"""	labels.extend(file_labels)
	filename = 'data/Voices-AWS/' + filename[20:-4] + '.wav'
	for i in range(len(starts)):
		split_audio(filename, identifiers[i], starts[i], ends[i])
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
		wavs.extend(identifiers[i] + ' /nobackup1c/users/mit-6345/kaldi_gpu/tools/sph2pipe_v2.5/sph2pipe -f wav data/Voices-AWS-utterances/merged/' + identifiers[i] + '.wav | ')

with open('data/Voices-AWS-utterances/text', 'w') as data_file:
	data_file.write("\n".join(utterances))

with open('data/Voices-AWS-utterances/wav.scp', 'w') as data_file:
        data_file.write("\n".join(wavs))

with open('data/Voices-AWS-utterances/labels', 'w') as data_file:
        data_file.write("\n".join(labels))
"""
with open('data/Voices-AWS-utterances/start', 'w') as data_file:
        data_file.write("\n".join(total_starts))

with open('data/Voices-AWS-utterances/end', 'w') as data_file:
        data_file.write("\n".join(total_ends))

print(len(utterances))
print(total_failed)
print(total_stutter)
print(total_inv)
print(female_stutter)
print(male_stutter)
print(female_inv)
print(male_inv)

durations = [a_i - b_i for a_i, b_i in zip(total_ends, total_starts)]

f, ax = plt.subplots(figsize=(9, 6))

ax.hist(durations, bins=40, align='mid', rwidth=.85)

plt.title('Distributions of Utterance Length')
plt.xlabel('Length of Utterance(s)')
plt.ylabel('Number of utterances')
f.savefig('data/audio_dur_dist.png')

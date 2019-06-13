import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('../data/Voices-AWS-google/confidences') as confidences:
	confidence_list = []

	for conf in confidences.readlines():
		file_name, confidence = conf.split(' ')
		confidence_list.append(float(confidence))
print(sum(confidence_list)/len(confidence_list))
f, ax = plt.subplots(figsize=(9, 6))
ax.hist(confidence_list, bins=40, align='mid', rwidth=.85)

plt.title('Confidence of Utterance Text')
plt.xlabel('Confidence of Utterance(s)')
plt.ylabel('Number of Utterances')
f.savefig('data/text_confidence_no_clean.png')

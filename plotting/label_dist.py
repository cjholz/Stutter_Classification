import os
import matplotlib.pyplot as plt

with open('data/Voices-AWS-segments-1/label') as labels:
    label_dict = {}
	num_labels = 0
    for label in labels.readlines():
        num_labels += 1
		file_name, label = label.split(' ')
       	if label in label_dict.keys():
			label_dict[label] +=1
		else:
			label_dict[label] = 1
	print(num_labels)
	print(label_dict)


	plt.bar([0, 1, 2, 3, 4, 5], [label_dict['0'], label_dict['1'], label_dict['2'], label_dict['3'], label_dict['4'], label_dict['5']])
	plt.show()

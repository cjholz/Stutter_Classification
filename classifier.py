import os
from random import shuffle
import csv
import numpy as np
import librosa.feature
from keras import models
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D, Flatten
from keras.callbacks import ModelCheckpoint
from keras import optimizers


def load_files(seg_len=1):

	files = []
	try:
		with open('data/Voices-AWS-segments-' + str(seg_len) + '/files.txt') as csv_file:
			csv_reader = csv.reader(csv_file)
			for line in csv_reader:
				files = line
	except Exception as e:
		dir_name = 'data/Voices-AWS-segments-' + str(seg_len) + '/'
		files = [os.path.join(dir_name, filename)
                	for dir_name, dirs, files in os.walk(dir_name)
                	for filename in files
                	if filename.endswith(".wav")]
		shuffle(files)
		with open('data/Voices-AWS-segments-' + str(seg_len) + '/files.txt', 'w') as data_file:
			data_file.write(",".join(files))

	Y_dict = {}
	with open('data/Voices-AWS-segments-'+ str(seg_len) + '/label', 'r') as label_file:
        	labels = label_file.readlines()
        	for line in labels:
                	line = line.split(" ")
                	Y_dict[line[0]] = int(line[1])
	print('Loaded files from: ' + 'data/Voices-AWS-segments-' + str(seg_len) + '/files.txt')
	return files, Y_dict


def get_features(files, Y_dict, use_chroma=False, use_mel_spec=False, use_tonnetz=False, use_contrast=False):
	X_all = []
	Y_all = []
	for filename in files:
		y, sr = librosa.load(filename)
		mfcc = librosa.feature.mfcc(y=y, sr=sr)
		chroma = None
		if use_tonnetz:
                	chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
		        tonnetz = librosa.feature.tonnetz(y=y, sr=sr, chroma=chroma)
                        mfcc = np.vstack([mfcc, tonnetz])
		if use_chroma:
			if not chroma:
				chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
			mfcc = np.vstack([mfcc, chroma])
		if use_mel_spec:
			mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
			mfcc = np.vstack([mfcc, mel_spec])
		if use_contrast:
			spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
                	mfcc = np.vstack([mfcc, spec_con])
		X_all.append(np.reshape(mfcc, (mfcc.shape[0], mfcc.shape[1], 1)))
		if seg_len == 1 or seg_len == 5:
			Y_all.append(Y_dict[filename[27:-4]])
		else:
			Y_all.append(Y_dict[filename[28:-4]])

	X = X_all[:int(len(X_all)*.6)]
	Y = Y_all[:int(len(X_all)*.6)]

	X_test = X_all[int(len(X_all)*.8):]
	Y_test = Y_all[int(len(X_all)*.8):]

	X_val = X_all[int(len(X_all)*.6):int(len(X_all)*.8)]
	Y_val = Y_all[int(len(X_all)*.6):int(len(X_all)*.8)]


	X = np.array(X)
	Y = np.array(Y)
	X_val = np.array(X_val)
	Y_val = np.array(Y_val)
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)

	return X, Y, X_val, Y_val, X_test, Y_test


def get_model(save_path='data/models/best_epoch_model.h5', dropout_rate=.5, epochs=50, batch_size=100, lr=.001, activation='softmax', initialization='truncated_normal', optimizer='adam'):
	nclass = 6
	inp = Input(shape=(X.shape[1], X.shape[2], 1))
	norm_inp = BatchNormalization()(inp)
	
	img_1 = Convolution2D(20, kernel_size=(2, 2), activation='relu', kernel_initializer=initialization)(norm_inp)
	img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
	img_1 = Dropout(rate=dropout_rate)(img_1)
	img_1 = Flatten()(img_1)
	dense_1 = BatchNormalization()(Dense(128, activation='relu')(img_1))
	dense_1 = BatchNormalization()(Dense(128, activation='relu')(dense_1))
	dense_1 = Dense(nclass, activation=activation, kernel_initializer=initialization)(dense_1)

	model = models.Model(inputs=inp, outputs=dense_1)

	checkpoint = ModelCheckpoint(save_path, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')	
	model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	print(model.summary())
	
	model.fit(X, Y, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
	
	return model

if __name__ == "__main__":
	use_contrast=True
	use_tonnetz = True

	seg_lens = [1, 5, 10, 15, 20]

	for seg_len in seg_lens:
		print('Starting' + str(seg_len))
		files, Y_dict = load_files(seg_len)

		X, Y, X_val, Y_val, X_test, Y_test = get_features(files, Y_dict, use_tonnetz=use_tonnetz, use_contrast=use_contrast)
		
		save_path = 'data/models/best_' + str(seg_len) +'_model.h5'
		get_model(save_path=save_path)

		model = models.load_model('data/models/best_' + str(seg_len) +'_model.h5')
		scores = model.evaluate(X_val, Y_val)

		pred = model.predict(X_test)
		pred = np.argmax(pred, axis=1)
		print(pred)
		print(Y_test)

		pred_stutter = [1 if p > 1 and p < 5 else 0 for p in pred]
		Y_stutter = [1 if p > 1 and p < 5 else 0 for p in Y_test]

		stutter_acc = [1 if pred_stutter[i] == Y_stutter[i] else 0 for i in range(len(pred))]

		pred_block = [1 if p == 0 or p == 4 else 0 for p in pred]
		Y_block = [1 if p == 0 or p == 4 else 0 for p in Y_test]

		block_acc = [1 if pred_block[i] == Y_block[i] else 0 for i in range(len(pred))]

		print(sum(stutter_acc)/float(len(stutter_acc)))
		print(sum(block_acc)/float(len(block_acc)))
		print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

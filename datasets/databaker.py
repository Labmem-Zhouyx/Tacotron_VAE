from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import os
import re
from datasets import audio


def build_from_path_CN(hparams, speaker_num, lan_num, input_dir, use_prosody, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the DataBaker dataset from a gven input path to given output directories
	(https://www.data-baker.com/open_source.html)

		DataBaker
			├── Wave             (dir of *.wav files)
			├── ProsodyLabeling  (dir of *.txt files, storing text/pronunciation/prosodic_structure information)
			└── PhoneLabeling    (dir of *.interval files, storing begin/end time of phonemes in Praat TextGrid format)

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- use_prosody: whether the prosodic structure labeling information will be used
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. This should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	content = _read_labels(os.path.join(input_dir, 'ProsodyLabeling'))
	num = int(len(content)//2)
	for idx in range(num):
		res = _parse_cn_prosody_label(content[idx*2], content[idx*2+1], use_prosody)
		if res is not None:
			basename, text = res
			wav_path = os.path.join(input_dir, 'Wave', '{}.wav'.format(basename))
			futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, speaker_num, lan_num, hparams)))

	return [future.result() for future in tqdm(futures) if future.result() is not None]

def build_from_path_EN(hparams, speaker_num, lan_num, input_dir, prefix, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):

  executor = ProcessPoolExecutor(max_workers=n_jobs)
  futures = []
  with open(os.path.join(input_dir, 'metadata.csv.txt'), 'r', encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('|')
      basename = prefix + parts[0]
      wav_path = os.path.join(input_dir, 'Wave', '{}.wav'.format(parts[0]))
      text = re.sub('[/%-]','',parts[1])
      futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, speaker_num, lan_num, hparams)))

  return [future.result() for future in tqdm(futures) if future.result() is not None]

def _read_labels(dir):
	"""
	Load all prosody labeling files from the directory
	"""

	# enumerate all *.txt files
	files = []
	# r=>root, d=>directories, f=>files
	for r, d, f in os.walk(dir):
		for item in f:
			if '.txt' in item:
				files.append(os.path.join(r, item))
	
	# load from all files
	labels = []
	for item in files:
		with open(item, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if line != '': labels.append(line)
	return labels

def _parse_en_label(text):
  # split into sub-terms
  regex = r"\s*([0-9]+)\.(.*)"
  sen_id, texts = text.split()
  # normalize the text
  texts = re.sub('[/\t\*;；？！，。]', '', texts)
  texts = texts.strip(' ')
  return sen_id, texts

def _parse_cn_prosody_label(text, pinyin, use_prosody=False):
	"""
	Parse label from text and pronunciation lines with prosodic structure labelings
	
	Input text:    100001 妈妈#1当时#1表示#3，儿子#1开心得#2像花儿#1一样#4。
	Input pinyin:  ma1 ma1 dang1 shi2 biao3 shi4 er2 zi5 kai1 xin1 de5 xiang4 huar1 yi2 yang4
	Return sen_id: 100001
	Return pinyin: ma1-ma1 dang1-shi2 biao3-shi4, er2-zi5 kai1-xin1-de5 / xiang4-huar1 yi2-yang4.

	Args:
		- text: Chinese characters with prosodic structure labeling, begin with sentence id for wav and interval file
		- pinyin: Pinyin pronunciations, with tone 1-5
		- use_prosody: Whether the prosodic structure labeling information will be used

	Returns:
		- (sen_id, pinyin&tag): latter contains pinyin string with optional prosodic structure tags
	"""

	text = text.strip()
	pinyin = pinyin.strip()
	if len(text) == 0:
		return None

	# remove punctuations
	text = re.sub('[“”、，。：；？！—…#（）]', '', text)

	# split into sub-terms
	sen_id, texts  = text.split()
	phones = pinyin.split()

	# prosody boundary tag (SYL: 音节, PWD: 韵律词, PPH: 韵律短语, IPH: 语调短语, SEN: 语句)
	SYL = '-'
	PWD = ' '
	PPH = ' / ' if use_prosody==True else ' '
	IPH = ', '
	SEN = '.'

	# parse details
	pinyin = ''
	i = 0 # texts index
	j = 0 # phones index
	b = 1 # left is boundary
	while i < len(texts):
		if texts[i].isdigit():
			if texts[i] == '1': pinyin += PWD  # Prosodic Word, 韵律词边界
			if texts[i] == '2': pinyin += PPH  # Prosodic Phrase, 韵律短语边界
			if texts[i] == '3': pinyin += IPH  # Intonation Phrase, 语调短语边界
			if texts[i] == '4': pinyin += SEN  # Sentence, 语句结束
			b  = 1
			i += 1
		elif texts[i]!='儿' or j==0 or not _is_erhua(phones[j-1][:-1]): # Chinese segment
			if b == 0: pinyin += SYL  # Syllable, 音节边界（韵律词内部）
			pinyin += phones[j]
			b  = 0
			i += 1
			j += 1
		else: # 儿化音
			i += 1

	return (sen_id, pinyin)

def _is_erhua(pinyin):
	"""
	Decide whether pinyin (without tone number) is retroflex (Erhua)
	"""
	if len(pinyin)<=1 or pinyin == 'er':
		return False
	elif pinyin[-1] == 'r':
		return True
	else:
		return False


def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, speaker_num, lan_num, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
			wav_path))
		return None

	#rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	#Get spectrogram from wav
	ret = audio.wav2spectrograms(wav, hparams)
	if ret is None:
		return None
	out, mel_spectrogram, linear_spectrogram, time_steps, mel_frames = ret

	# Write the spectrogram and audio to disk
	audio_filename = 'audio-{}.npy'.format(index)
	mel_filename = 'mel-{}.npy'.format(index)
	linear_filename = 'linear-{}.npy'.format(index)
	np.save(os.path.join(wav_dir, audio_filename), out.astype(np.float32), allow_pickle=False)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
	np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example
	return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text, speaker_num, lan_num)

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import os
import re
from datasets import audio


def build_from_path(hparams, speaker_num, lan_num, input_dir, prefix, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
  """
  Preprocesses the TH-CoSS dataset from a gven input path to given output directories

    TH-CoSS
      ├── wav     (dir of *.wav, *.lab files, storing wav and lab files)
      └── *.txt   (prompt files, storing text and pinyin prompts)

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
  with open(os.path.join(input_dir, 'main.csv.txt'), 'r', encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('|')
      basename = prefix + parts[0]
      wav_path = os.path.join(input_dir, 'main', '{}.wav'.format(parts[0]))
      text = parts[2].replace('/','')
      futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, speaker_num, lan_num, hparams)))

  return [future.result() for future in tqdm(futures) if future.result() is not None]


def build_from_path_simple(hparams, speaker_num, lan_num, input_dir, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
  """
  Preprocesses the TH-CoSS dataset from a gven input path to given output directories

    TH-CoSS
      ├── wav     (dir of *.wav, *.lab files, storing wav and lab files)
      └── *.txt   (prompt files, storing text and pinyin prompts)

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
  with open(os.path.join(input_dir, 'metadata.csv.txt'), 'r', encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('|')
      basename = parts[0]
      wav_path = os.path.join(input_dir, 'wave', '{}.wav'.format(parts[0]))
      text = parts[2].replace('/','')
      futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, speaker_num, lan_num, hparams)))

  return [future.result() for future in tqdm(futures) if future.result() is not None]



def _read_labels(path):
  """
  Load the text and pinyin prompts from the file
  """
  labels = []
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if line != '': labels.append(line)
  return labels

def _parse_cn_prosody_label(text, pinyin, use_prosody=False):
  """
  Parse label from text and pronunciation lines with prosodic structure labelings

  Input text:    1.	/为临帖/他*还|远游|西*安|碑林/龙门|石窟/泰山|*摩崖|石刻/./
  Input pinyin:  wei4 lin2 tie4 ta1 hai2 yuan3 you2 xi1 an1 bei1 lin2 long2 men2 shi2 ku1 tai4 shan1 mo2 ya2 shi2 ke4
  Return sen_id: 1
  Return pinyin: wei4-lin2-tie4 / ta1-hai2 yuan3-you2 xi1-an1 bei1-lin2 / long2-men2 shi2-ku1 / tai4-shan1 mo2-ya2 shi2-ke4.

  Args:
    - text: Chinese characters with prosodic structure labeling, begin with sentence id for wav and label file
    - pinyin: Pinyin pronunciations, with tone 1-5
    - use_prosody: Whether the prosodic structure labeling information will be used

  Returns:
    - (sen_id, pinyin&tag): latter contains pinyin string with optional prosodic structure tags
  """

  # split into sub-terms
  regex = r"\s*([0-9]+)\.(.*)"
  match = re.match(regex, text)
  if not match:
    return None

  # split into sub-terms
  sen_id = int(match.group(1))
  texts = match.group(2)
  phones = pinyin.strip().split()

  # normalize the text
  texts = re.sub('[ \t\*;?!,.；？！，。]', '', texts)
  texts = texts.replace('//', '/')
  texts = texts.replace('||', '|')
  if texts[0]  in ['/', '|']: texts = texts[1:]
  if texts[-1] in ['/', '|']: texts = texts[:-1]+'.'

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
    if texts[i] in ['|', '/', ',', '.']:
      if texts[i] == '|': pinyin += PWD  # Prosodic Word, 韵律词边界
      if texts[i] == '/': pinyin += PPH  # Prosodic Phrase, 韵律短语边界
      if texts[i] == ',': pinyin += IPH  # Intonation Phrase, 语调短语边界
      if texts[i] == '.': pinyin += SEN  # Sentence, 语句结束
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
  pinyin = pinyin.replace('E', 'ev') # 特殊发音E->ev

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

  This writes the mel scale spectogram to disk and return a tuple to write to the train.txt file

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
    print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
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

import argparse
import os
from warnings import warn
from time import sleep

import tensorflow as tf
from hparams import hparams
from infolog import log
from tacotron.synthesize import tacotron_synthesize
from zh_cn import G2P

def prepare_run(args, weight):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	run_name = args.name or args.tacotron_name or args.model
	taco_checkpoint = os.path.join('Tacotron_VAE/logs-' + run_name + weight , 'taco_' + args.checkpoint)
	return taco_checkpoint, modified_hp

def get_sentences(args, websen=None):
	if args.text_list == '':
		a_sentences = hparams.sentences
	elif args.text_list == 'web':
		a_sentences = websen

	else:
		with open(args.text_list, 'rb') as f:
			a_sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
	print(a_sentences)
	sentences=[]
	speaker_labels=[]
	language_labels=[]
	for i, line in enumerate(a_sentences):
		line = line.strip('\t\r\n').split('|')
		sentences.append(line[0])
		speaker_labels.append(int(line[1]))
		#0 denotes English, 1 denotes Chinese
		language_labels.append(int(line[2]))
	g2p=G2P()
	#sentences = [g2p.convert(i) for i in sentences]
	return sentences, speaker_labels, language_labels



def main(websen=None, weight=''):

	accepted_modes = ['eval', 'synthesis', 'live']
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
	parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
	parser.add_argument('--wavenet_name', help='Name of logging directory of WaveNet. If trained separately')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
	parser.add_argument('--mels_dir', default='tacotron_output/eval/', help='folder to contain mels to synthesize audio from using the Wavenet')
	parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--mode', default='eval', help='mode of run: can be one of {}'.format(accepted_modes))
	parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
	parser.add_argument('--text_list', default='web', help='Text file contains list of texts to be synthesized. Valid if mode=eval')
	parser.add_argument('--speaker_id', default=None, help='Defines the speakers ids to use when running standalone Wavenet on a folder of mels. this variable must be a comma-separated list of ids')
	args = parser.parse_args(args=[])

	accepted_models = ['Tacotron']

	if args.model not in accepted_models:
		raise ValueError('please enter a valid model to synthesize with: {}'.format(accepted_models))

	if args.mode not in accepted_modes:
		raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))

	if args.GTA not in ('True', 'False'):
		raise ValueError('GTA option must be either True or False')

	taco_checkpoint, hparams = prepare_run(args, weight)
	sentences, speaker_labels, language_labels = get_sentences(args, websen)
	print(sentences)
	if args.model == 'Tacotron':
		_ = tacotron_synthesize(args, hparams, taco_checkpoint, sentences, speaker_labels, language_labels)
	else:
		raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))


if __name__ == '__main__':
	main()

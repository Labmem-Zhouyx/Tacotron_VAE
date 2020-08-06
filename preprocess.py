import argparse
import os
from multiprocessing import cpu_count

from hparams import hparams
from tqdm import tqdm
from datasets import ljspeech
from datasets import databaker
from datasets import thcoss
from datasets import Huawei

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[4]) for m in metadata])
	timesteps = sum([int(m[3]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))


def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--dataset', default='all')
	parser.add_argument('--output', default='training_data')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	modified_hp = hparams.parse(args.hparams)
	
	# Prepare directories
	in_dir  = os.path.join(args.base_dir, args.dataset)
	out_dir = os.path.join(args.base_dir, args.output)
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	lin_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(lin_dir, exist_ok=True)
	
	# Process dataset
	metadata = []
	if args.dataset == 'all':
		use_prosody = False

		in_dir = os.path.join(args.base_dir, 'LJSpeech-1.1')
		print('processing LJSpeech-1.1.../n')
		metadata_1 = ljspeech.build_from_path(modified_hp, 0, 0, in_dir, mel_dir, lin_dir, wav_dir, args.n_jobs, tqdm=tqdm)

		in_dir = os.path.join(args.base_dir, 'DataBaker')
		print('processing DataBaker.../n')
		metadata_2 = databaker.build_from_path_CN(modified_hp, 1, 1, in_dir, use_prosody, mel_dir, lin_dir, wav_dir, args.n_jobs,tqdm=tqdm)
		
		in_dir = os.path.join(args.base_dir, 'TTS.HUawei.zhcmn.F.Deng')
		print('processing TTS.HUawei.zhcmn.F.Deng.../n')
		metadata_3 = Huawei.build_from_path_CN(modified_hp, 2, 1, in_dir, use_prosody, mel_dir, lin_dir, wav_dir, args.n_jobs, tqdm=tqdm)

		in_dir = os.path.join(args.base_dir, 'TTS.Huawei.enus.F.XuYue')
		print('processing TTS.Huawei.enus.F.XuYue.../n')
		metadata_4 = Huawei.build_from_path_EN(modified_hp, 3, 0, in_dir, mel_dir, lin_dir, wav_dir, args.n_jobs, tqdm=tqdm)
		
		in_dir = os.path.join(args.base_dir, 'TTS.THCoSS.zhcmn.F.M/TH-CoSS/data/03FR00')
		print('processing TTS.THCoSS.zhcmn.F.M 03FR00.../n')
		metadata_5 = thcoss.build_from_path(modified_hp, 4, 1, in_dir, 'a', mel_dir, lin_dir, wav_dir, args.n_jobs,
										  tqdm=tqdm)

		in_dir = os.path.join(args.base_dir, 'TTS.THCoSS.zhcmn.F.M/TH-CoSS/data/03MR00')
		print('processing TTS.THCoSS.zhcmn.F.M 03MR00.../n')
		metadata_6 = thcoss.build_from_path(modified_hp, 5, 1, in_dir, 'b',  mel_dir, lin_dir, wav_dir, args.n_jobs,
											tqdm=tqdm)
											 
		in_dir = os.path.join(args.base_dir, 'TTS.Pachira.zhcmn.enus.F.DB1/zh-cmn')
		print('processing TTS.Pachira.zhcmn.enus.F.DB1/zh-cmn.../n')
		metadata_7 = thcoss.build_from_path_simple(modified_hp, 6, 1, in_dir,  mel_dir, lin_dir, wav_dir, args.n_jobs,
											tqdm=tqdm)

		in_dir = os.path.join(args.base_dir, 'TTS.DataBaker.enus.M.DB1')
		print('processing TTS.DataBaker.enus.M.DB1.../n')
		metadata_8 = databaker.build_from_path_EN(modified_hp, 7, 0, in_dir, 'x', mel_dir, lin_dir, wav_dir, args.n_jobs,
												   tqdm=tqdm)

		in_dir = os.path.join(args.base_dir, 'TTS.DataBaker.enus.F.DB1')
		print('processing TTS.DataBaker.enus.F.DB1.../n')
		metadata_9 = databaker.build_from_path_EN(modified_hp, 8, 0, in_dir, 'y', mel_dir, lin_dir, wav_dir, args.n_jobs,
												   tqdm=tqdm)

		in_dir = os.path.join(args.base_dir, 'TTS.DataBaker.enus.F.DB2')
		print('processing TTS.DataBaker.enus.F.DB2.../n')
		metadata_10 = databaker.build_from_path_EN(modified_hp, 9, 0, in_dir, 'z', mel_dir, lin_dir, wav_dir, args.n_jobs,
												   tqdm=tqdm)

		metadata = metadata_1 + metadata_2 + metadata_3 + metadata_4 + metadata_5 + metadata_6 + metadata_7 + metadata_8 + metadata_9 + metadata_10

	elif args.dataset == 'LJSpeech-1.1':
		metadata = ljspeech.build_from_path(modified_hp, in_dir, mel_dir, lin_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	elif args.dataset == 'DataBaker':
		use_prosody = False
		metadata = databaker.build_from_path_CN(modified_hp, in_dir, use_prosody, mel_dir, lin_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	elif args.dataset == 'THCoSS':
		use_prosody = True
		metadata = thcoss.build_from_path(modified_hp, in_dir, use_prosody, mel_dir, lin_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	else:
		raise ValueError('Unsupported dataset provided: {} '.format(args.dataset))
	
	# Write metadata to 'train.txt' for training
	write_metadata(metadata, out_dir)



if __name__ == '__main__':
	main()

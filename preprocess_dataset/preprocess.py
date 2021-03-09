import argparse
import os
from multiprocessing import cpu_count

# from tqdm import tqdm
import numpy as np
import mel

from concurrent.futures import ProcessPoolExecutor
from functools import partial



# in
original_txt_path = '/ceph/home/hujk17/Tuned-EarSpeech/preprocess_dataset/meta.txt'

# out
out_txt_path = ['train.txt', 'val.txt', 'test.txt']




def _process_utterance(audio_path, text, npy_path):
	if os.path.exists(audio_path):
		mel_spectrogram, _linear_spectrogram, out = mel.wav2mel(audio_path)
		time_steps = len(out)
		mel_frames = mel_spectrogram.shape[0]
	else:
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(audio_path))
		return None

	#    /ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/spk-004996.npy
	#->  /ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/mel-004996.npy
	mel_same_with_npy_path = npy_path.replace('spk', 'mel').split('.')[0] + '-mel.npy'
	np.save(mel_same_with_npy_path, mel_spectrogram.T, allow_pickle=False)
	# print(mel_same_with_npy_path)
	# print(mel_spectrogram.shape, mel_spectrogram)
	# Return a tuple describing this training example
	return (mel_same_with_npy_path, npy_path, time_steps, mel_frames, text)



def main():
	executor = ProcessPoolExecutor(max_workers=cpu_count())
	futures = []
	with open(original_txt_path, encoding='utf-8') as f:
		f_list = f.readlines()
		for x in f_list:
			audio_path, txt, npy_path = x.strip().split('|')
			futures.append(executor.submit(partial(_process_utterance, audio_path, txt, npy_path)))
			# break

		metadata = [future.result() for future in futures if future.result() is not None]
	
	len_max = min(65573, len(metadata))
	start = [0, 60000, 65000]
	endd = [59999, 64999, len_max]
	for k in range(3):
		with open(out_txt_path[k], 'w', encoding='utf-8') as f:
			for i in range(start[k], endd[k]):
				f.write('|'.join([str(x) for x in metadata[i]]) + '\n')
				# break
		# break
	
	print('Write {} utterances'.format(len(metadata)))
	# print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	# print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))


if __name__ == '__main__':
	main()

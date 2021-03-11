from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
import numpy as np
import torch
import os
from preprocess_dataset.gl import write_wav, mel2wav



enc_model_fpath = None # 张阳版本滴还没有
syn_model_fpath = '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/synthesizer/saved_models/FaPig/FaPig_55k.pt'
voc_model_fpath = None # 先用 gl


input_meta = ['/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB0005/mel-SSB00050119-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB0005/spk-SSB00050119.npy|81000|406|ti3 yu4 zong3 ju2 gai1 gan4 de5 rang4 ti3 yu4 zong3 ju2 gan4',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB0005/mel-SSB00050117-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB0005/spk-SSB00050117.npy|97800|490|you3 liu4 shi2 jia1 gong1 si1 wei4 xin1 xing1 chan3 ye4 shang4 shi4 gong1 si1',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB0323/mel-SSB03230294-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB0323/spk-SSB03230294.npy|80000|401|xing2 cheng2 ji4 you3 fen1 gong1 you4 xiang1 hu4 pei4 he2 de5 jian1 guan3 ji1 zhi4',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB0323/mel-SSB03230295-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB0323/spk-SSB03230295.npy|57200|287|ping2 jun1 shou4 jia4 si4 bai2 jiu3 shi2 jiu2 mei3 yuan2',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB1711/mel-SSB17110171-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB1711/spk-SSB17110171.npy|83400|418|nan2 guai4 ta1 zi4 chao2 dan1 xin1 zhe4 ge4 yang4 zi5 xing2 chu1 jie1 hui4 bei4 ren2 da3',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB1711/mel-SSB17110174-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB1711/spk-SSB17110174.npy|80600|404|qi1 huo4 ying2 ye4 bu4 shu4 liang4 shi2 xian4 le5 ping2 wen2 you3 xu4 zeng1 zhang3',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/mel-002126-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/spk-002126.npy|79200|397|lao3 bo2 jue2 shuo1 ge1 zi5 nen2 zai4 ta3 shang5 xie1 xi5 na4 qun2 huo3 ji1 ze2 zhu4 zai4 di4 yi1 cen2',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/mel-002131-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/spk-002131.npy|48400|243|san1 liang4 yue4 ye3 che1 zai4 cong2 lin2 dao4 lu4 shang5 gao1 su4 lue4 guo4',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB1711/mel-SSB17110187-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB1711/spk-SSB17110187.npy|59600|299|er2 wo3 men5 zhe4 ge4 tuan2 dui4 jin1 nian2 xin1 zu3 cheng2',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB1711/mel-SSB17110188-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy/SSB1711/spk-SSB17110188.npy|61400|308|gao1 dang4 jiao4 che1 xu1 jia1 jiu3 shi2 qi1 hao4 qi4 you2',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S1-female-5000/mel-000196-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S1-female-5000/spk-000196.npy|42600|214|si1 ding1 ni2 ba3 suo2 you3 zhe4 yi2 qie4 ao2 zai4 guo1 li3',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S1-female-5000/mel-000456-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S1-female-5000/spk-000456.npy|79400|398|jia2 dao3 zai4 ye3 bu2 yong4 wei4 ying2 zhan3 qiu2 ren2 le5 ye3 zhong1 yu2 ke2 yi3 shui4 jiao4 le5 wo3 hai2 ting3 kai1 xin1 de5',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S1-female-5000/mel-004726-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S1-female-5000/spk-004726.npy|52600|264|zhun3 bei4 zhi1 fu4 yi4 qian1 wan4 zui4 duo1 yi4 qian1 wu2 bai3 wan4 mei3 yuan2',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/mel-001556-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/spk-001556.npy|89000|446|shi4 bing1 shuo1 dao4 ta1 de5 zui2 jiao3 lu4 chu1 le5 yi4 si1 jian1 xiao4 ta1 xing1 li3 fei1 chang2 qing1 chu3 zhe4 jian4 shi4',
'/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/mel-001576-mel.npy|/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/spk-001576.npy|56000|281|hai2 you3 wang2 you3 wen4 ai4 si1 bo2 ni3 wei4 shen2 me5 zhe4 me5 pa4 zhong1 guo2',]

input_text = ['ti3 yu4 zong3 ju2 gai1 gan4 de5 rang4 ti3 yu4 zong3 ju2 gan4',
'you3 liu4 shi2 jia1 gong1 si1 wei4 xin1 xing1 chan3 ye4 shang4 shi4 gong1 si1',
'xing2 cheng2 ji4 you3 fen1 gong1 you4 xiang1 hu4 pei4 he2 de5 jian1 guan3 ji1 zhi4',
'wo3 xi3 huan1 ni3 zhou1 cui4 bao4 zao4 ge1',
'xie4 xie4 da4 jia1 lin2 ting1 yu3 yin1 he2 cheng2 shi2 yan4',]

if __name__ == '__main__':

    # if args.cpu:
    #     # Hide GPUs from Pytorch to force CPU processing
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ""

        
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    # encoder.load_model(enc_model_fpath)
    synthesizer = Synthesizer(syn_model_fpath)
    # vocoder.load_model(voc_model_fpath)
    
    
    
    for i, _speaker_info in enumerate(input_meta):
        for j, text in enumerate(input_text):
            # speaker embedding
            speaker_info = _speaker_info.strip().split('|')
            # embed = np.load(speaker_info[1])
            GE2E_path = speaker_info[1].replace('.npy', '-GE2E.npy')
            GE2E_embed = np.load(GE2E_path)
            print('speaker embedding shape:', GE2E_embed.shape)
            print("Created the embedding")
            mel_reference = np.load(speaker_info[0]).T
            
            
            
            # # If seed is specified, reset torch seed and force synthesizer reload
            # if args.seed is not None:
            #     torch.manual_seed(args.seed)
            #     synthesizer = Synthesizer(args.syn_model_fpath)

            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [GE2E_embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            mels = synthesizer.synthesize_spectrograms(texts, embeds)
            mel = mels[0]
            mel = mel.T

            print('reference:', mel_reference.shape)
            
            print("Created the mel spectrogram")


            os.makedirs('log_FaPig_GE2E_syn_wavs', exist_ok=True)
            _wav_pre = mel2wav(mel, wav_name_path=os.path.join('log_FaPig_GE2E_syn_wavs', 'spk_' + str(i) + '_' + str(j) + '_pre_GE2E.wav'))
            _wav_target = mel2wav(mel_reference, wav_name_path=os.path.join('log_FaPig_GE2E_syn_wavs', 'spk_' + str(i) + '_' + str(j) + '_reference_GE2E.wav'))

            # break
        # break

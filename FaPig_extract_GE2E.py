import os
from pathlib import Path
from synthesizer.preprocess import embed_utterance



# [1] all meta path
# in_path
metas_path = ['/ceph/dataset/AISHELL-3_denoise_dereverb_/train/content.txt']
father_path = '/ceph/dataset/M2VoC'



for _root, dirs, _files in os.walk(father_path): 
    # print('hh', dirs)
    for now_dir in dirs:
        # print('yyyyyyyy:', now_dir)
        if now_dir != 'wav':
            now_abs_dir = os.path.join(father_path, now_dir)
            for __root, __dirs, txts in os.walk(now_abs_dir):  
                for txt in txts:
                    if os.path.splitext(txt)[1] == '.txt':  
                        metas_path.append(os.path.join(_root, os.path.join(txt.split('.')[0], txt)))

print(metas_path)




class speech(object):
     def __init__(self,):
        self.audio_path = None
        self.txt = None
        self.speaker_npy_path = None



res = []

with open(metas_path[0], 'r', encoding='utf-8') as f:
    f_list = f.readlines()
    for x in f_list:
        # SSB00050001.wav	广 guang3 州 zhou1 女 nv3 大 da4 学 xue2
        # print('origi:', x)
        x = x.strip().split()
        # print(x)

        # audio_path
        name = x[0]
        audio_path = os.path.join(os.path.join('/ceph/dataset/AISHELL-3_denoise_dereverb_/train/wav', name.split('.')[0][:-4]), name)
        if os.path.exists(audio_path) is False:
            continue

        # txt
        txt_list = []
        for i in range(2, len(x), 2):
            txt_list.append(x[i])
        txt = ' '.join(txt_list)
        # print(txt)

        # speaker_npy_path
        npy_path = os.path.join(os.path.join('/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy', name.split('.')[0][:-4]), 'spk-' + name.split('.')[0] + '.npy')
        if os.path.exists(npy_path) is False:
            continue


        # add speech
        a = speech()
        a.audio_path = audio_path
        a.txt = txt
        a.speaker_npy_path = npy_path
        res.append(a)
        # break


for no in range(2, len(metas_path)):
    now_file = metas_path[no]
    if '/ceph/dataset/M2VoC/TSV-Track1-S2-female-Anchor-100/TSV-Track1-S2-female-Anchor-100.txt' == now_file:
        continue
    if '/ceph/dataset/M2VoC/TSV-Track1-S1-male-Sales-100/TSV-Track1-S1-male-Sales-100.txt' == now_file:
        continue
    with open(now_file, 'r', encoding='utf-8') as f:
        f_list = f.readlines()
        for i in range(0, len(f_list), 5):
            # 000001
            # 场外基金的投资者可以在基金开放日的交易时间内到销售网点修改分红方式。
            # chang3 wai4 ji1 jin1 de5 tou2 zi1 zhe3 ke3 yi3 zai4 ji1 jin1 kai1 fang4 ri4 de5 jiao1 yi4 shi2 jian1 nei4 dao4 xiao1 shou4 wang3 dian3 xiu1 gai3 fen1 hong2 fang1 shi4
            # 场外#1 基金的#1 投资者#3 可以在#1 基金#1 开放日的#1 交易#1 时间内#1 到#1 销售#1 网点#1 修改#1 分红#1 方式#5

            # audio_path
            main_dir = '/'.join(now_file.split('/')[:-1])
            name = f_list[i].strip()
            # print(main_dir)
            audio_path = os.path.join(main_dir, os.path.join('wavs', name + '.wav'))
            if os.path.exists(audio_path) is False:
                continue
            
            # txt
            txt = f_list[i + 2].strip().replace('[] ', '')
            # print(txt)

            # speaker_npy_path
            npy_dir = now_file.split('/')[-1].split('.')[0]
            npy_path = os.path.join(os.path.join('/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy', npy_dir), 'spk-' + name + '.npy')
            if os.path.exists(npy_path) is False:
                continue


            # add speech
            a = speech()
            a.audio_path = audio_path
            a.txt = txt
            a.speaker_npy_path = npy_path
            res.append(a)
            # break
    # break



# 在 res 列表中算 embedding
for x in res:
    audio_path = x.audio_path

    npy = x.speaker_npy_path
    GE2E_npy = npy.split('.')[0] + '-GE2E.npy'
    embed_utterance((audio_path, GE2E_npy), encoder_model_fpath=Path('encoder/saved_models/pretrained.pt'))
    # print('now has:', GE2E_npy)
    # break


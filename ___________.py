#!python
# /storage/LabJob/Projects/Data/CovoST4/cv-corpus-6.1-2020-12-11/en/clips//common_voice_en_19551053.mp3
import torch, torchaudio, torchaudio.transforms as T
from pathlib import Path
import pandas as pd

dfs = dict(
    train=pd.read_csv('tmp/train.tsv', sep='\t'),
    dev=pd.read_csv('tmp/dev.tsv', sep='\t'),
    test=pd.read_csv('tmp/test.tsv', sep='\t'),
)

CV_ROOT = Path("/storage/LabJob/Projects/Data/CovoST4/""cv-corpus-6.1-2020-12-11/en/clips/")

MODEL_SAMPLE_RATE: int = 16000

RESAMPLER_DICT = {
    48000: T.Resample(  # for Common Voice
        48000, MODEL_SAMPLE_RATE, 
        dtype=torch.float32),
    MODEL_SAMPLE_RATE: lambda x: x,  # identity
}

def resample_audio(
        audio_tensor: torch.Tensor, 
        original_sample_rate: int,
        target_sample_rate: int = MODEL_SAMPLE_RATE,
    ):
    """ Resample music to target_sample_rate. """
    if original_sample_rate not in RESAMPLER_DICT:
        RESAMPLER_DICT[original_sample_rate] = T.Resample(
            original_sample_rate, target_sample_rate, 
            dtype=audio_tensor.dtype)
    return RESAMPLER_DICT[original_sample_rate](audio_tensor), target_sample_rate
    

def resample_and_dump(audio_src: str, audio_tgt: str, target_sample_rate: int = MODEL_SAMPLE_RATE, int16: bool = True):
    ''' Resample audio and save it to audio_tgt. '''
    assert Path(audio_src).is_file()
    audio_tensor, original_sample_rate = torchaudio.load(audio_src)
    resampled, target_sample_rate = resample_audio(audio_tensor, original_sample_rate, target_sample_rate)
    if int16:
        resampled = (resampled * 32768).short()
        torchaudio.save(audio_tgt, resampled, target_sample_rate, encoding='PCM_S', bits_per_sample=16)
    else:
        torchaudio.save(audio_tgt, resampled, target_sample_rate)
    return audio_tensor.size(-1), resampled.size(-1)

# audio_path: Path = CV_ROOT / audio_path

# rr, srr = torchaudio.load('rr.wav')
# rr2, sr2 = torchaudio.load('rr2.wav')
# rr3, sr3 = torchaudio.load('rr3.wav')

from tqdm import tqdm
for SPLIT in [
    # 'train', 
    'dev', 'test']:
    with open(f'tmp/{SPLIT}_length.tsv', 'w') as fout:
        NEW_ROOT = f'/storage/LabJob/Projects/PipelineBuild/wavs/{SPLIT}'
        if not Path(NEW_ROOT).is_dir():
            Path(NEW_ROOT).mkdir(parents=True)
        print(NEW_ROOT, file=fout) 
        for mp3 in tqdm(dfs[SPLIT]["path"], desc=f"Resampling {SPLIT}..."):
            basename = Path(mp3).stem
            _, tgt_length = resample_and_dump(
                CV_ROOT / mp3,
                Path(NEW_ROOT) / (basename + '.wav'),
            )
            print(f"{basename}.wav", tgt_length, sep='\t', file=fout)

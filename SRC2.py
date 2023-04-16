import torchaudio, torchaudio.transforms as T

model_SR = 16000

with open('/storage/LabJob/Projects/PipelineBuild/EricDataset/dst/audiopaths.tsv', 'r') as f:
    root, *filenames = f.read().strip().split('\n')
    filenames = [line.split('\t')[0] for line in filenames]


# mkdir wavs
tgtdir = 'wavs'

resample_rate = model_SR
lengths = []
# outs = []
with open('newdir.tsv', 'w') as fout:
    print('/', file=fout)  # TOFIX
    for filename in filenames:
        waveform, sample_rate = torchaudio.load(f"{root}/{filename}")
        resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
        resampled_waveform = resampler(waveform)
        # outs.append(resampled_waveform)
        lengths.append(resampled_waveform.size(-1))
        tgtname = f'{tgtdir}/{filename}.wav'
        torchaudio.save(tgtname, resampled_waveform, resample_rate)
        print(tgtname, resampled_waveform.size(-1), sep='\t', file=fout)


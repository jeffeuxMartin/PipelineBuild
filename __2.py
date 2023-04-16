#!python

from fairseq_utils.dump_hubert_feature import (
    HubertFeatureReader, 
    dump_feature, 
    get_path_iterator,
)
# from fairseq_utils import dump_hubert_feature
from fairseq_utils.kmeans_quantizer import KMeansQuantizer

print(HubertFeatureReader)
print(KMeansQuantizer)
# exit(0)

# reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
# generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
# dump_feature(reader, generator, num, split, nshard, rank, feat_dir)

import torch; from typing import Optional, Callable, Dict; import torchaudio


def get_streams(
    waveform,
    # speaker,
    dense_model,
    quantizer_model,
    # f0_normalizer,
    # f0_quantizer,
    # need_f0,
    # deduplicate,
    # f0_code_ratio,
):
    if waveform.ndim > 1:
        waveform = waveform.mean(0)

    dense_features = dense_model(waveform)
    units = quantizer_model(dense_features)

    collunits, durations = torch.unique_consecutive(units, return_counts=True)

    # if need_f0:
    #     f0 = get_f0(waveform.cpu().numpy())
    #     f0 = torch.from_numpy(f0).float()

    #     if f0_normalizer:
    #         f0 = f0_normalizer(f0, speaker)
    #     tol = 5 * f0_code_ratio
    #     f0 = align_f0_to_durations(f0, durations, f0_code_ratio, tol)
    #     if f0_quantizer:
    #         f0 = f0_quantizer(f0)
    # else:
    #     f0 = None

    return units, (collunits, durations), dense_features

class SpeechEncoder(torch.nn.Module):
    """SpeechEncoder encodes speech into streams of (pseudo-)units, unit durations,
    and, optionally, F0.
    """

    def __init__(
        self,
        upstream_ckpt_path: str,
        kmeans_ckpt_path: str,
        layer: int,
        max_chunk: int,
        
        # 
        # dense_model: torch.nn.Module,
        # quantizer_model: torch.nn.Module,
        # deduplicate: bool,
        # add_bos_eos: bool = False,
        # need_f0: bool = True,
        # f0_normalizer: Optional[Callable] = None,
        # f0_quantizer: Optional[Callable] = None,
    ):
        """Builds a SpeechEncoder instance. SpeechEncoder encodes speech into streams of (pseudo-)units, unit durations,
        and, optionally, F0.

        Args:
            dense_model (torch.nn.Module): Dense module used to represent the audio
            quantizer_model (torch.nn.Module): Quantize module that converts dense representation into discrete tokens
            deduplicate (bool): if set, run-length encoding is applied so that repeated tokens are deduplicated
                and duration channel contains the number of repeats of the token.
            add_bos_eos (bool, optional): if set, each token sequences will be prepended with a special token (bos)
                and appended with another special token (eos).
            need_f0 (bool, optional): whether F0 stream should be returned. Estimating F0 is computationally heavy,
                consider disabling it if not needed.
            f0_normalizer (Optional[Callable], optional): A callback that allows F0 normalization (e.g., per-speaker)
            f0_quantizer (Optional[Callable], optional): F0 quantization module
        """
        super().__init__()
        self.dense_model = HubertFeatureReader(upstream_ckpt_path, layer, max_chunk)
        self.quantizer_model = KMeansQuantizer(kmeans_ckpt_path)
        
        
        self.RESAMPLER_DICT = {
            48000: torchaudio.transforms.Resample(  # for Common Voice
                48000, self.expected_sample_rate, 
                dtype=torch.float32),
            self.expected_sample_rate: lambda x: x,  # identity
        }

        

        # self.add_bos_eos = add_bos_eos
        # self.need_f0 = need_f0
        # self.f0_normalizer = f0_normalizer
        # self.f0_quantizer = f0_quantizer

        # self.unit_vocab_size = self.quantizer_model.vocab_size

        # self.register_buffer(
        #     "bos", torch.tensor([self.unit_vocab_size], dtype=torch.int)
        # )
        # self.register_buffer(
        #     "eos", torch.tensor([self.unit_vocab_size + 1], dtype=torch.int)
        # )

        # self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def expected_sample_rate(self) -> int:
        """
        int: sample rate expected by the underlying dense model
        """
        return self.dense_model.expected_sample_rate
    
    @property
    def device(self) -> torch.device:
        """
        Returns:
            torch.device: device where the speech encoder resides
        """
        return self._float_tensor.device

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int: vocabulary size used for the unit stream (NB: not counting bos/eos/pad tokens)
        """
        return self.quantizer_model.vocab_size

    @property
    def code_hop_size(self) -> int:
        """
        Returns:
            int: hop step size of the dense model
        """
        return self.dense_model.code_hop_size

    @property
    def expected_sample_rate(self) -> int:
        """
        int: sample rate expected by the underlying dense model
        """
        return self.dense_model.expected_sample_rate

    def resample(
            self,
            waveform: torch.Tensor, 
            input_sample_rate: int,
        ):
        """ Resample music to target_sample_rate. """
        if input_sample_rate == self.expected_sample_rate:
            return waveform
        if input_sample_rate not in self.RESAMPLER_DICT:
            self.RESAMPLER_DICT[input_sample_rate] = torchaudio.transforms.Resample(
                input_sample_rate, self.expected_sample_rate, 
                dtype=waveform.dtype)
        return self.RESAMPLER_DICT[input_sample_rate](waveform)
        

    def forward(
        self, waveform: torch.Tensor, speaker: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Encodes a raw waveform tensor into two or three aligned & synchronised streams: pseudo-unit (token),
        duration, and pitch aka F0 streams. F0 is only provided if the SpeechEncoder instance
        was initialized with `requires_f0=True`.

        Args:
            waveform (torch.Tensor): audio to be encoded
            speaker (Optional[str], optional): speaker id to be passed to the F0 normalizer.
            Can be safely ignored if no F0 stream is requested or no per-speaker F0 normalizer
            provided.

        Returns:
            Dict[str, torch.Tensor]: dictionary with the following keys:
             * "units": contains an int tensor with the unit stream,
             * "durations": duration of each unit, measured in frames,
             * "dense": dense encoding of the audio, as provided by the underlying dense model,
             * "f0": F0 stream - only returned if `requires_f0=True` was set in the constructor.
        """
        units, durations, f0, dense_features = get_streams(
            waveform,
            speaker,
            self.dense_model,
            self.quantizer_model,
            self.f0_normalizer,
            self.f0_quantizer,
            self.need_f0,
            self.deduplicate,
            self.f0_code_ratio,
        )

        # if self.add_bos_eos:
        #     units, durations, f0, dense_features = wrap_bos_eos(
        #         units, durations, f0, dense_features, self.bos, self.eos
        #     )

        item = {
            "units": units.to(self.device),
            "durations": durations.to(self.device),
            "dense": dense_features,
        }
        if f0 is not None:
            item["f0"] = f0

        return item


class HubertFeatureReader(nn.Module):
    def __init__(self, upstream_ckpt_path, layer, max_chunk):
        super().__init__()
        self.upstream_ckpt_path = upstream_ckpt_path
        self.layer = layer
        self.max_chunk = max_chunk
        self.expected_sample_rate = 16000
        self.code_hop_size = 160

        self.model = torch.hub.load(
            "pytorch/fairseq",
            "hubert_base",
            checkpoint_file=self.upstream_ckpt_path,
            feature_extract=True,
        )
        self.model.eval()
        self.model.to("cuda")

    def forward(self, waveform):
        # waveform = waveform.to("cuda")
        with torch.no_grad():
            x = self.model.extract_features(waveform)
            x = x[self.layer]
            x = x.transpose(1, 2)
            x = x.reshape(x.shape[0], -1)
            x = x.transpose(0, 1)
            x = x.unsqueeze(0)
            return x
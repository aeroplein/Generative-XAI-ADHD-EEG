import numpy as np
from scipy.signal import butter, lfilter, iirnotch, spectrogram, resample_poly
from typing import Optional
from pathlib import Path

from utils.logger import LoggerHelper
from d4pm import d4PMDenoiser

class PreprocessorHelper:

    def __init__(self, fs:float, lowcut: float=0.5, highcut:float=50.0,
                 order:int=5, use_d4pm:bool=False,
                 device:str="cpu", dp4m_weights_path: Optional[str]=None):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.use_d4pm = use_d4pm

        self.logger = LoggerHelper.get_logger()

        if self.use_d4pm:
            self.logger_info("D4PM-Based Denoising is enabled.")
            self.d4pm = d4PMDenoiser(
                device=device,
                weights_path=dp4m_weights_path
            )
        else:
            self.d4pm = None


    def harmonize_sampling_rate(self, data: np.ndarray, original_fs:float) -> np.ndarray:
        if original_fs == self.fs:
            return data
        
        self.logger.info(f"Resampling data from {original_fs} Hz to {self.fs} Hz")
        return resample_poly(data, up=int(self.fs), down=int(original_fs), axis=-1)


    def _butterworth_bandpass(self):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        return butter(self.order, [low, high], btype="band")
    
    def _apply_classical_filters(self, data: np.ndarray):
        b, a = self._butterworth_bandpass()
        filtered = lfilter(b, a, data, axis=-1)

        notch_freq = 50.0
        quaility_factor = 30.0
        b_notch, a_notch = iirnotch(notch_freq / (0.5 * self.fs), quaility_factor)
        return lfilter(b_notch, a_notch, filtered, axis=-1)
    
    def _apply_d4pm(self, data: np.ndarray, artifact_type: str, snr_db:float) -> np.ndarray:
        if self.d4pm is None:
            return data
        
        self.logger.info(f"Applying D4PM, SNR={snr_db} dB")
        lambda_snr = 10 ** (-snr_db / 20)

        if data.ndim ==1:
            return self.d4pm.denoise(data, artifact_type, lambda_snr)
        
        output = np.zeros_like(data)
        for channel in range(data.shape[0]):
            output[channel] = self.d4pm.denoise(data[channel], artifact_type, lambda_snr)
        return output
    
    def preprocess(self, data:np.ndarray, artifact_type: Optional[str]=None, snr_db: float=0.0) -> np.ndarray:
        if self.use_d4pm and artifact_type is not None:
            data = self._apply_d4pm(data, artifact_type, snr_db)
        return self._apply_classical_filters(data)
    
    def segment_data(self, data:np.ndarray, segment_length:float, overlap:float=0.5)->np.ndarray:
        samples = int(segment_length * self.fs)
        step = int(samples * (1 - overlap))

        segments = []

        for start in range(0, data.shape[-1] - samples + 1, step):
            segments.append(data[..., start:start + samples])
        return np.array(segments)
    
    def compute_spectrogram(self, data:np.ndarray):
        f, t, Sxx = spectrogram(data, fs=self.fs, nperseg=128, noverlap=112, nfft=256, axis=-1)
        return np.log1p(Sxx)
    # we are moving from time domain to the time frequency domain
    # delta and theta are loud, high amplitude, high frequencies like gamma is quiet.
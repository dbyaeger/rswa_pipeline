import numpy as np
from scipy.signal import filtfilt, firwin, resample_poly, freqz
import numba


class Analyzer1:
    """
    features are calculated via:
    1 optionally high-pass filtered at 20Hz (for EMG)
    2 rectified
    3 resampled to 10Hz (from 100Hz)
    """
    def __init__(self, fs):
        self.fs = fs
        B = firwin(65, 20, fs=fs, pass_zero=False)  # high-pass filter at 20Hz for EMG signals
        if 0:  # inspect frequency response
            from matplotlib import pyplot as plt
            w, H = freqz(B, [1.])
            plt.plot(w * mtt.fs / 2 / np.pi, 20 * np.log10(np.abs(H)))  # in dB
            plt.show()
        self.B_emg = B
        B = firwin(65, 5, fs=fs)  # low-pass filter at 5Hz, for resampling to Fs=10Hz
        if 0:  # inspect frequency response
            from matplotlib import pyplot as plt
            w, H = freqz(B, [1.])
            plt.plot(w * mtt.fs / 2 / np.pi, 20 * np.log10(np.abs(H)))  # in dB
            plt.show()
        self.B_down = B

    def analyze(self, x: np.ndarray, emg_filtering: bool) -> np.ndarray:
        if emg_filtering:
            x = filtfilt(self.B_emg, [1.], x)
        x = np.abs(x)
        x = filtfilt(self.B_down, [1.], x)
        if self.fs == 100:
            x = resample_poly(x, 1, 10)
        else:
            AssertionError('for now fs is assumed to be 100')
        x[x < 0] = 0  # because values could be slightly less than zero due to the downsampling/smoothing filter
        return x  # this is now at 10 Hz


@numba.jit(nopython=True, cache=True)  # need polymorphism here
def segment(x, nsize, nrate):
    """return full (non-padded) segments"""
    if len(x) < nsize:
        F = 0
    else:
        F = (len(x) - nsize) // nrate + 1  # the number of full frames
    X = np.empty((F, nsize), dtype=x.dtype)
    a = 0
    for f in range(F):
        X[f, :] = x[a:(a + nsize)]
        a += nrate
    return X


@numba.jit(nopython=True, cache=True)  # need polymorphism here
def _analyze(x: np.ndarray, nsize: int, nrate: int) -> (np.ndarray, np.ndarray):
    # segment and feature extraction per segment in a loop to conserve memory
    if len(x) < nsize:
        F = 0
    else:
        F = (len(x) - nsize) // nrate + 1  # the number of full frames
    x_max = np.empty(F)
    x_rms = np.empty(F)
    a = 0
    for f in range(F):
        b = a + nsize
        # feature calculation
        x_max[f] = max(x[a:b])
        x_rms[f] = (x[a:b] ** 2).mean() ** 0.5
        # advance
        a += nrate
    time = np.arange(F) * nrate + nsize // 2
    return time, x_max, x_rms


class Analyzer2:
    def __init__(self, fs, inspect_frequency_response=False):
        self.fs = fs
        B = firwin(65, 20, fs=fs, pass_zero=False)  # high-pass filter at 20Hz for EMG signals
        if inspect_frequency_response:
            from matplotlib import pyplot as plt
            w, H = freqz(B, [1.])
            plt.plot(w * fs / 2 / np.pi, 20 * np.log10(np.abs(H)))  # in dB
            plt.show()
        self.B_emg = B

    def analyze(self, x: np.ndarray, emg_filtering: bool,
                nsize: int = 10, nrate: int = 10) -> (np.ndarray, np.ndarray):
        if emg_filtering:
            x = filtfilt(self.B_emg, [1.], x)  # not supported by numba
        x = np.abs(x)  # rectification
        time, x_max, x_rms = _analyze(x, nsize, nrate)
        return time, np.c_[x_max, x_rms]

    # def analyze_old(self, x: np.ndarray, emg_filtering: bool,
    #             nsize: int = 10, nrate: int = 10) -> (np.ndarray, np.ndarray):
    #     if emg_filtering:
    #         x = filtfilt(self.B_emg, [1.], x)
    #     x = np.abs(x)  # rectification
    #     X = segment(x, nsize, nrate)
    #     x1 = X.max(axis=1)
    #     x2 = (X ** 2).mean(axis=1) ** 0.5  # RMS-energy
    #     time = np.arange(X.shape[0]) * nrate + nsize // 2
    #     return time, np.c_[x1, x2]


def main():
    import pickle
    from view import convert_p_to_mtt
    file_path = '../data/example/XAXVDJYND7Q6JTK_2.p'

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    mtt = convert_p_to_mtt(data)
    signal = 'Snore'
    #, 'Chin', 'L Leg', 'R Leg']  # signals of interest
    #signals = ['EOG-L', 'EOG-R']  # signals of interest
    #signals = ['O2-A1', 'C4-A1', 'F4-A1', 'C3-A2', 'F3-A2', 'O1-A2']  # signals of interest

    nsize = 100
    nrate = 100

    analyzer = Analyzer2(mtt.fs)
    x = mtt[signal].value[:, 0]
    t, y = analyzer.analyze(x, emg_filtering=False, nsize=nsize, nrate=nrate)
    # t2, y2 = analyzer.analyze_old(x, emg_filtering=False, nsize=nsize, nrate=nrate)
    # assert (t == t2).all()
    # assert np.allclose(y, y2)

    import matplotlib.pyplot as plt
    tx = np.arange(len(x))
    plt.plot(tx, x, alpha=0.3)
    plt.plot(t, y, '.')  # available values
    y2 = np.array([np.interp(tx, t, y) for y in y.T]).T  # interpolating y to *all* values of x, one dim at a time
    plt.plot(tx, y2)  # interpolated values
    plt.show()


if __name__ == '__main__':
    main()

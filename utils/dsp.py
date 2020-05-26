import numpy as np
from scipy import signal


def design_filter(fs, hi):
    B = signal.firwin(65, hi, fs=fs, pass_zero=True)  # band-pass filter
    A = np.array([1.])
    # if 0:  # inspect frequency response
    #     w, H = signal.freqz(B, A)
    #     plt.plot(w * fs / 2 / np.pi, 20 * np.log10(np.abs(H)))  # in dB
    #     plt.show()
    return B, A


def filter_and_resample(x, B, A, fs_old=200, fs_new=100):
    # TODO: rewrite this more generically, but for now:
    assert fs_old == 200, 'original fs must be 200Hz'
    assert fs_new == 100, 'new sampling frequency must be 100Hz'

    x = signal.filtfilt(B, A, x)  # zero-phase/delay filtering
    # resample from 200 to 100 Hz by decimating with factor of 2
    I = 1
    D = 2
    #fs_new = fs_old
    #fs_new *= I
    #fs_new /= D
    return signal.resample_poly(x, I, D)


def reduce_ecg(sig, ecg, fs):
    pass


def main_filter():
    from pathlib import Path
    import matplotlib.pyplot as plt

    import openxdf

    record = 'XVZ2FFAG886BVQ0'
    data_dir = Path("../../data/example")
    xdf_ext = '.xdf'
    data_ext = '.nkamp'

    file_path = Path(data_dir / record)

    xdf = openxdf.OpenXDF(file_path.with_suffix(xdf_ext))
    signals = openxdf.Signal(xdf, file_path.with_suffix(data_ext))

    signal_dict = signals.read_file(["ECG", "Chin"])

    # grab a couple of minutes
    sig1 = signal_dict["Chin"][:120].ravel()
    ecg1 = signal_dict["ECG"][:120].ravel()
    fs1 = 200
    fs2 = 100
    B, A = design_filter(fs1, hi=fs2/2)
    sig2 = filter_and_resample(sig1, B, A, fs1, fs2)
    ecg2 = filter_and_resample(ecg1, B, A, fs1, fs2)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(np.arange(len(ecg1)) / fs1, ecg1)
    ax[0].plot(np.arange(len(ecg2)) / fs2, ecg2)
    ax[0].set_ylabel('ecg')
    ax[1].plot(np.arange(len(sig1)) / fs1, sig1)
    ax[1].plot(np.arange(len(sig2)) / fs2, sig2)
    ax[1].set_ylabel('sig')

    plt.show()


def main_reduce_ecg():
    import matplotlib.pyplot as plt
    from pathlib import Path
    import pickle

    record = 'XVZ2FFAG88SECZT_3'

    data_dir = Path("../../data/example")
    file_path = Path(data_dir / record).with_suffix('.p')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    fs = data['signals']['ECG'].shape[1]
    sig = data['signals']['L Leg'].ravel()
    ecg = data['signals']['ECG'].ravel()

    cln = sig + .1 * ecg

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(np.arange(len(ecg)) / fs, ecg)
    ax[0].set_ylabel('ecg')
    ax[1].plot(np.arange(len(sig)) / fs, sig)
    ax[1].set_ylabel('sig')
    ax[2].plot(np.arange(len(cln)) / fs, cln)
    ax[2].set_ylabel('cln')
    #ax[2].axis(xmin=0, xmax=3)


    plt.show()


if __name__ == "__main__":
    main_filter()

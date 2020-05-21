from openxdf.xdf import OpenXDF
from openxdf.signal import Signal
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_data(xdf_path, signal_path):
    xdf = OpenXDF(xdf_path)
    signal = Signal(xdf, signal_path)
    return xdf, signal

def load_xdf(xdf_path):
    xdf = OpenXDF(xdf_path)
    return xdf

def get_stages(xdf, scorer="Dennis"):
    staging = xdf.dataframe(epochs=True, events=False)
    staging = staging[staging["Scorer"] == scorer]
    epoch_number = np.array(staging['EpochNumber'])
    stages = np.array(staging['Stage'])
    return epoch_number, stages

def get_cpap_start_epoch(xdf, scorer="Dennis"):
    "Returns the epoch at which cpap, ipap, or epap field is first non-zero"
    df = xdf.dataframe(epochs=True, events=False)
    df = df[df['Scorer']=="Dennis"]
    found_cpap = False
    for field in df.columns:
        if 'pap' in field:
            channel = field
            found_cpap = True
            break
    assert found_cpap, f"Cpap not in dataframe: {df.columns}"
    cpap_epochs =  df[df[channel] > 0]['EpochNumber'].values
    if len(cpap_epochs) > 0:
        return cpap_epochs.min()
    else:
        return None

def get_notes(file,xdf):
    """Returns notes by technicians in a dictionary keyed by time in seconds from
    the start of the study"""
    # Get notes
    notes = xdf._parse(file,deidentify=True)['xdf:NoteLog']['xdf:Note']
    
    # Get start time
    start = xdf.start_time
    
    # convert to dictionary of time: note
    note_dict = {(datetime.strptime(note['xdf:Time'][:-9], "%Y-%m-%dT%H:%M:%S.%f")-start).seconds:note['xdf:NoteText'] for note in notes}
    return note_dict

def parse_notes(notes, epoch_length=30):
    """Looks through the notes and finds the earliest note containing items
    related to cpap.
    """
    words = ['bilevel','cpap','bipap','ipap','epap','mask']
    earliest = np.float('inf')
    for time in notes:
        if notes[time] is None:
            continue
        event = notes[time].lower().strip()
        if event == 'split night start':
            earliest = time
            break
        else:
            for word in words:
                if word in event:
                    earliest = min(earliest, time)
    # convert to epoch
    if earliest < np.float('inf'):
        #print(f'earliest note: {notes[earliest]}')
        return earliest//epoch_length
    else: return None
        
def select_rem_epochs(signal, scorer="Dennis"):
    staging = signal._xdf.dataframe(epochs=False, events=True)
    staging = staging[staging["Scorer"] == scorer].reset_index(drop=True)
    return staging[staging["Stage"] == "R"].reset_index(drop=True)

def select_rswa_events(signal, scorer="Dennis"):
    staging = signal._xdf.dataframe(epochs=False, events=True)
    staging = staging[staging["Scorer"] == scorer].reset_index(drop=True)
    staging = staging[
        staging["CEName"].isnull() | staging["CEName"].isin(["RSWA_T", "RSWA_P"])
        ]
    staging = staging[staging["CEName"].isin(["RSWA_T", "RSWA_P"])]
    return staging


def select_epoch(epoch, channel, signals_dict):
    start = (epoch - 1) * 30
    end = start + 30
    period = signals_dict[channel][start:end]
    return np.concatenate(period)


def plot_epoch(epoch, signals_dict, scoring_df):
    plot_signals = {}
    for channel in signals_dict.keys():
        plot_signals[channel] = select_epoch(epoch, channel, signals_dict)

    ## Plot channels
    fig, ax = plt.subplots(1)
    min_point, max_point = 0, 0
    for key in plot_signals.keys():
        len_key = len(plot_signals[key])
        x_time_axis = [i / (len_key / 30) for i in range(0, len_key)]
        min_point = min(min_point, np.min(plot_signals[key]))
        max_point = max(max_point, np.max(plot_signals[key]))
        ax.plot(x_time_axis, plot_signals[key], label=key)
        ax.legend()

    ## Add event annotations
    events = scoring_df[scoring_df["EpochNumber"] == epoch]
    if events.size:
        for indx, row in events.iterrows():
            if row["CEName"] == "RSWA_P":
                ax.axvline(x=row["EpochTime"], linestyle="--", color="black")

            if row["CEName"] == "RSWA_T":
                x_start = float(row["EpochTime"])
                y_start = min_point
                width = float(row["Duration"])
                height = max_point - min_point
                rect = Rectangle((x_start, y_start), width, height, linewidth=1, fill=True, alpha=0.5, facecolor="grey")
                ax.add_patch(rect)
    plt.show()


def plot_signal(data):
    sig = data["signals"]["Chin"]
    n_seconds, sampling_rate = sig.shape
    fig, ax = plt.subplots(1, figsize=(14, 10))
    min_point, max_point = 0, 0
    x_time_axis = list(i / sampling_rate for i in range(n_seconds * sampling_rate))
    min_point = min(min_point, sig.min())
    max_point = max(max_point, sig.max())
    ax.plot(x_time_axis, sig.ravel(), label="Chin")
    ax.set_xlabel("seconds")
    ax.set_ylabel("signal")
    ax.legend()

    ## Add event annotations
    events = data["events"]
    if events:
        for start_time, end_time, etype in events:
            start_time -= data["start_time"]
            end_time -= data["start_time"]
            print(start_time, end_time, etype)
            if etype == "RSWA_P":
                ax.axvline(x=start_time, linestyle="--", color="black")

            if etype == "RSWA_T":
                print("RSWA_T")
                x_start = start_time
                y_start = min_point
                width = float(end_time - start_time)
                height = max_point - min_point
                rect = Rectangle((x_start, y_start), width, height, 
                                 linewidth=1, fill=True, alpha=0.5, 
                                 facecolor="grey")
                ax.add_patch(rect)
    plt.show()


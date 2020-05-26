import os
import sys
import pdb
import time
import pickle
#from timeout_decorator import timeout_decorator
from artifactreduce.artifact_reduce import ArtifactReducer
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from collections import defaultdict, Counter
from utils.open_xdf_helpers import (load_data, 
                                    select_rem_epochs, 
                                    select_rswa_events, 
                                    get_cpap_start_epoch,
                                    get_notes,
                                    parse_notes)
from utils.dsp import design_filter, filter_and_resample


class DataProcessor:

    def __init__(self, input_path, output_path, channel_list=["Chin", "L Leg", "R Leg"], 
                 min_time_hours=4, min_rem_epochs=10, min_rem_subsequence_len=1,
                 artifact_reduce=False, 
                 path_to_polysmith_db = '/Users/danielyaeger/Documents/Modules/sleep-research-ml/data/supplemental/Polysmith_DataBase_ML_forPhil.csv',):
        """
        Takes in xdf and nkamp files corresponding to one complete study and writes
        to disk individual signal files, one for each REM subsequence in the corresponding study
        INPUT:
            input_path (string): path to folder containing xdf and correspnding nkamp files
        OUTPUT:
            (None): writes processed files to disk with the following format:
            
                {
                    "ID":ID,
                    "study_start_time":study_start_time,
                    "staging":[(start_time, end_time, stage) for nrem and rem],
                    "rswa_events":[(start_time, end_time, type) w.r.t. rsw_events],
                    "signals":{"signal_name":[raw_signal] w.r.t. all signals}
                }
        """
        print(f"processing data from {input_path}")
        self.path = Path(input_path)
        assert self.path.is_dir(), f"{input_path} is not a valid path"

        self.channel_list = channel_list
        self.min_time_hours = min_time_hours
        self.min_rem_epochs = min_rem_epochs
        self.min_rem_subsequence_len = min_rem_subsequence_len
        self.artifact_reduce =  artifact_reduce
        if 'ECG' not in self.channel_list: self.channel_list.append('ECG')
        
        # Collect study metadata
        if not isinstance(path_to_polysmith_db,Path): path_to_polysmith_db = Path(path_to_polysmith_db)
        self.study_meta_data = pd.read_csv(path_to_polysmith_db, usecols=['Sex','Age (yrs)','RecType','RecordNo'])
        self.psg_studies = list(self.study_meta_data[self.study_meta_data['RecType'] == 'PSG']['RecordNo'].values)
        self.split_studies = list(self.study_meta_data[self.study_meta_data['RecType'].isin(['SPLIT'])]['RecordNo'].values)

        # define path to write out processed data and create dir if it does not exist
        self.output_data_path = self.path.joinpath(output_path)
        if not self.output_data_path.is_dir():
            self.output_data_path.mkdir()

        # keep track of IDs that have already been processed in case a run dies part way through
        if self.output_data_path.joinpath("history.p").is_file():
            with self.output_data_path.joinpath("history.p").open("rb") as f_in:
                self.history = pickle.load(f_in)
        else:
            self.history = {}

        self.files = [f.parent.joinpath(f.stem) for f in self.path.iterdir() 
                      if f.suffix == '.xdf']

    #@timeout_decorator.timeout(300)
    def _process_helper(self, f):
        tstart = time.time()
        EPOCH_LEN = 30
        ID = f.stem
        if ID in self.history:
            print(f"{ID} already processed")
            return

        # load xdf and signal data and pull dataframe
        xdf, signal = load_data(str(f) + ".xdf", str(f) + ".nkamp")
        df = xdf.dataframe()
        df = df[df["Scorer"] == "Dennis"]
        final_epoch = df["EpochNumber"].max()
        df_rem = df[df["Stage"] == "R"]
        study_start_time = xdf.start_time
        signal_dict = signal.read_file(self.channel_list)

        # insure min rem epochs are predent
        if len(df_rem) < self.min_rem_epochs:
            e = "EXIT: min number of REM epochs not present in study"
            print(e)
            self.history[ID] = e
            return
        
        # if study is less than min_time OR no REM epochs exist we will exclude the study
        if signal_dict[self.channel_list[0]].shape[0] / 60 / 60 < self.min_time_hours:
            e = "EXIT: study is shorter than min allowable time"
            print(e)
            self.history[ID] = e
            return
        
        # Check if last epoch complete
        end_time, sampling_rate = signal_dict[self.channel_list[0]].shape
        
        # split study, restrict to 60 minutes before cpap machine turned on
        if ID in self.split_studies:
            field_cpap_start = get_cpap_start_epoch(xdf)
            notes = get_notes(str(self.path.joinpath(ID + '.xdf')),xdf)
            notes_cpap_start = parse_notes(notes)
            if field_cpap_start is None and notes_cpap_start is not None:
                final_epoch = notes_cpap_start - 60
            elif field_cpap_start is not None and notes_cpap_start is None:
                final_epoch = field_cpap_start - 60
            elif field_cpap_start is not None and notes_cpap_start is not None:
                final_epoch = min(notes_cpap_start,field_cpap_start) - 60
            assert final_epoch > 20, f"Cpap machine turns on during epoch {final_epoch + 1}!"
        
        if end_time < final_epoch*EPOCH_LEN:
                final_epoch = len(signal_dict[self.channel_list[0]])//EPOCH_LEN
            
        end_time = final_epoch*EPOCH_LEN
        
        # Filter for final epoch
        df = df[df["EpochNumber"] <= final_epoch]
        df_rem = df_rem[df_rem["EpochNumber"] <= final_epoch]

        # check data load latency (this seems to be the main bottleneck and may be optimized in the openxdf package)
        print(f"{time.time() - tstart:.2f} seconds elapsed for data load")

        # organize REM epochs into lists of contiguous epochs for downstream sliding window formatting
        rem_epochs = self._create_contiguous_rem_epochs(df_rem["EpochNumber"].unique())

        # ignore 0th REM subsequence flag
        ignore_0th_subseq = False

        # only include REM epochs
        for idx, epoch_set in enumerate(rem_epochs): 

            # exclude rem sebsequences that are too short
            if len(epoch_set) < self.min_rem_subsequence_len:
                continue

            start_epoch_rem = epoch_set[0]
            start_epoch = self._find_start_epoch(df = df, rem_epochs = rem_epochs, idx = idx)
            end_epoch = epoch_set[-1]

            nrem_epochs = self._extract_nrem_epochs(df, epoch_set, start_epoch, end_epoch)

            staging = []

            if not nrem_epochs and idx == 0:
                ignore_0th_subseq = True
                continue

            if nrem_epochs:
                staging.append(((min(nrem_epochs) - 1) * EPOCH_LEN, max(nrem_epochs) * EPOCH_LEN, 'N'))
                start_time_n_rem = (nrem_epochs[0] - 1) * EPOCH_LEN
                end_time_n_rem = nrem_epochs[-1] * EPOCH_LEN
                if self.artifact_reduce:
                    filter_dict = {}
                    for channel in ["Chin", "L Leg", "R Leg"]:
                        B, A = design_filter(sampling_rate, 50)
                        ecg_sig = signal_dict['ECG'][start_time_n_rem:end_time_n_rem].ravel()
                        emg_sig = signal_dict[channel][start_time_n_rem:end_time_n_rem].ravel()
                        filter_dict[channel] = ArtifactReducer()
                        filter_dict[channel].fit(ecg = filter_and_resample(ecg_sig, B, A),
                                   emg = filter_and_resample(emg_sig, B, A))
                
            # need to subtract 1 as EpochNumber starts at 1    
            start_time_rem = (start_epoch_rem - 1) * EPOCH_LEN
            end_time_rem = end_epoch * EPOCH_LEN # we don't subtract 1 here in order to include the last epoch

            staging.append((start_time_rem, end_time_rem, 'R'))
                
            df_filtered = df_rem[df_rem["EpochNumber"].isin(epoch_set)]
            
            rswa_events = self._extract_rswa_events(df_filtered)

            output = {
                "ID":ID,
                "study_start_time":study_start_time,
                "staging":staging,
                "rswa_events":rswa_events,
                "signals":{}
                }
            
            for c in self.channel_list:
                B, A = design_filter(sampling_rate, 50)
                if nrem_epochs:
                    emg_sig = np.vstack((signal_dict[c][start_time_n_rem:end_time_n_rem],signal_dict[c][start_time_rem :end_time_rem])).ravel()
                    ecg_sig = np.vstack((signal_dict['ECG'][start_time_n_rem:end_time_n_rem],signal_dict['ECG'][start_time_rem :end_time_rem])).ravel()
                    emg_sig = filter_and_resample(emg_sig, B, A)
                    ecg_sig = filter_and_resample(ecg_sig, B, A)
                    if self.artifact_reduce:
                        if c in ["Chin", "L Leg", "R Leg"]:
                            [emg_filt, noise_reduction] = filter_dict[channel].process_arrays(ecg = ecg_sig, emg = emg_sig)
                            emg_sig = emg_filt.reshape(-1,100)
                    else:
                        emg_sig = emg_sig.reshape(-1,100)
                else:
                    emg_sig = signal_dict[c][start_time_rem :end_time_rem].ravel()
                    ecg_sig = signal_dict['ECG'][start_time_rem :end_time_rem].ravel()
                    emg_sig = filter_and_resample(emg_sig, B, A)
                    ecg_sig = filter_and_resample(ecg_sig, B, A)
                    if self.artifact_reduce:
                        if c in ["Chin", "L Leg", "R Leg"]:
                            [emg_filt, noise_reduction] = filter_dict[channel].process_arrays(ecg = ecg_sig, emg = emg_sig)
                            emg_sig = emg_filt.reshape(-1,100)
                    else:
                        emg_sig = emg_sig.reshape(-1,100)

                output["signals"][c] = emg_sig

            if idx == 0:
                assert len(nrem_epochs) > 0, f"For ID {ID} and REM subsequence {idx}, no non-REM baseline found!"
            
            assert output['signals']['Chin'].shape[0] == sum([t[1] - t[0] for t in output['staging']]), \
            f"For ID {ID}, mismatch between signal length and staging. Staging: {output['staging']} and signal length: {output['signals']['Chin'].shape[0]}"

            if ignore_0th_subseq: 
                idx -= 1

            # write out a file for each id + epoch combination with a ID - signal pair for each channel
            file_name = f"{ID}_{idx}.p"
            with self.output_data_path.joinpath(file_name).open("wb") as f_out:
                pickle.dump(output, f_out)

        self.history[ID] = "processing successful"
        print(f"{ID} proccessing complete ({time.time() - tstart:.2f} seconds)")


    @staticmethod
    def _extract_nrem_epochs(df, epoch_set, start_epoch, end_epoch):
        """
        Collect up to 4 epochs of not-rem sleep (N) preceding each REM subsequence
        Throw an error if a N stage is found within an REM subsequence (bug)
        """
        nrem_epochs = []
        nrem_stages = ['1', '2', '3']
        epoch_set_s = set(epoch_set)
        for epoch, stage in df[df["EpochNumber"].isin(range(start_epoch, end_epoch + 1))].loc[:, ["EpochNumber", "Stage"]].drop_duplicates().values:
            if stage != 'R' and epoch in epoch_set_s:
                raise ValueError("non-REM stage found in REM subsequence (potential bug in _create_contiguous_rem_epochs method")
            elif stage in nrem_stages:
                # only include stages from that are consecutive, otherwise move up start time accordingly
                if not nrem_epochs or epoch == (nrem_epochs[-1] + 1):
                    nrem_epochs.append(epoch)
                else:
                    start_epoch += len(nrem_epochs)
                    nrem_epochs = [epoch]
            elif stage not in nrem_stages + ['R']:
                # exclude all other stages (e.g. W)
                # if we exclude a preceding epoch we need to push up the start by 1
                start_epoch += 1
                # print(epoch, stage, "excluded", "start_epoch", start_epoch)
            else:
                continue
        return nrem_epochs


    def _extract_apnia_hypopnia_events(self, xdf):
        """
        Extract all apnia and hypopnia events from a study with respect to a subset of epochs
        """
        apnia_events = xdf.events["Dennis"].get("Apneas") or []
        hypopnia_events = xdf.events["Dennis"].get("Hypopneas") or []
        apnia_hypopnia_events = []
        for i, event in enumerate(apnia_events + hypopnia_events):
            t, d = event["Time"], event["Duration"]
            event_type = 'A' if i < len(apnia_events) else 'H'
            tt = (self.format_datetime(t) - xdf.start_time).total_seconds()
            apnia_hypopnia_events.append((round(tt, 6), round(tt + float(d), 6), event_type))
        return apnia_hypopnia_events


    @staticmethod
    def _extract_rswa_events(df):
        """
        Create a list of events where each event is represented as (start_time, end_time, event_type)
        If a study does not contain any RSWA events then the corresponding dataframe will not contain CEName
        """
        rswa_events = []
        if "CEName" in df.columns:
            for n, t, event_name, d in df.loc[:, ["EpochNumber", "EpochTime", "CEName", "Duration"]].dropna().values:
                # EpochTime and Duration are both in seconds so start_time and end_time will be in seconds
                # Need to use EpochNumber - 1 as before since indexing starts at 1
                if event_name[-1] not in ['T', 'P']:
                    continue
                event_start = (n - 1) * 30 + round(float(t), 6)
                event_end = event_start
                if d:
                    event_end += float(d)
                rswa_events.append((event_start, event_end, event_name[-1]))
        return rswa_events
    
    
    @staticmethod
    def _create_contiguous_rem_epochs(epoch_list):
        if len(epoch_list) == 1:
            return epoch_list
        epoch_list = sorted(epoch_list)
        out = []
        curr = [epoch_list[0]]
        for epoch_next in epoch_list[1:]:
            if epoch_next == curr[-1] + 1:
                curr.append(epoch_next)
            else:
                out.append(curr)
                curr = [epoch_next]
        out.append(curr)
        return out


    @staticmethod
    def _find_start_epoch(df: pd.DataFrame, rem_epochs: list, idx: int):
        """
        Finds the start epoch, or the first epoch in a string of up to four
        consecutive non-REM epochs preceding a REM subsequence. Returns the 
        epoch number.
        """
        start_epoch_rem = rem_epochs[idx][0]
        if idx > 0:
            return start_epoch_rem - min(4, start_epoch_rem - (rem_epochs[idx - 1][-1] + 1))
        else:
            # This code works fine with _extract_nrem_epochs because _extract_nrem_epochs returns the nrem_epochs, not the start_epoch
            nrem_epochs = df[df["EpochNumber"].isin(range(start_epoch_rem-30, start_epoch_rem)) & (df['Stage'] != 'W')]["EpochNumber"].drop_duplicates().values
            nrem_epoch_groups = DataProcessor._create_contiguous_rem_epochs(nrem_epochs)
            return nrem_epoch_groups[-1][-4:][0]
                

    @staticmethod
    def format_datetime(s):
        """
        Convert a date string of the format "%Y-%m-%dT%H:%M:%S.%f" to a datetime object
        """
        # The milliseconds can be an issue so need to be truncated in cases
        pre, suff = s.split(".")
        suff = suff[:6]
        sf = ".".join([pre, suff])
        return datetime.strptime(sf, "%Y-%m-%dT%H:%M:%S.%f")

    
    def process_data(self):
        n_tot = len(self.files)
        for i, f in enumerate(self.files):
            print(f"{i + 1:3d} / {n_tot} {f.stem}")
            try:
                self._process_helper(f)

            except Exception as e:
                self.history[f.stem] = e
                print("ERROR", e)
                continue

        # write out history file to output data dir
        with self.output_data_path.joinpath("history.p").open("wb") as f_out:
            pickle.dump(self.history, f_out)


if __name__ == "__main__":
    data_path = "/Volumes/TOSHIBA EXT/training_data"

    output_path = "/Users/danielyaeger/Documents/NO_artifact_reduced_emg2"

    data_processor = DataProcessor(data_path,output_path)
    data_processor.process_data()
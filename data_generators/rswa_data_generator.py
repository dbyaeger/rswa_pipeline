from tensorflow import keras
import numpy as np
import pickle
from utils.ts_img_conversion import GAF
from sklearn.preprocessing import MinMaxScaler
from skimage import transform as T
from collections import OrderedDict
from scipy.signal import resample
from analysis import Analyzer1 as Analyzer
from pathlib import Path
import pdb

class DataGeneratorAllWindows(keras.utils.Sequence):

    def __init__(self, data_path, apnea_dict_path, batch_size=128, window_size=10,
                 channel_list=["Chin", "L Leg", "R Leg"],
                 n_classes=3, stride=1, shuffle=True,
                 eps=0.5, pos_prob=0.3, mode="train"):

        mode = mode.lower()
        assert mode in ["train", "cv", "val", "validation", "test"],\
            "mode must be train, test or val"

        if not isinstance(data_path, Path): data_path = Path(data_path)
        
        if not isinstance(apnea_dict_path, Path): apnea_dict_path = Path(apnea_dict_path)
        
        with apnea_dict_path.open('rb') as fa:
            self.apnea_dict = pickle.load(fa)['predictions']

        self.data_path = data_path
        self.batch_size = batch_size
        self.channel_list = channel_list
        self.n_channels = len(channel_list)
        self.n_classes = n_classes
        self.window_size = window_size
        self.shuffle = True if mode != "test" else False
        self.DOWNSAMPLED_RATE = 10
        self.data_format = None

        self.dim = (window_size * self.DOWNSAMPLED_RATE, self.n_channels)

        with data_path.joinpath("data_partition.p").open("rb") as f_in:
            partition = pickle.load(f_in)
        if mode == "train":
            self.list_IDs = list(partition["train"])
        elif mode in ["cv", "val", "validation"]:
            self.list_IDs = list(partition["val"]) if "val" in partition else list(partition["cv"])
        else:
            self.list_IDs = list(partition["test"])

        # fail dafe incase data_partition contains files that do not exist
        id_set = set([f.stem for f in data_path.iterdir()])
        for ID in self.list_IDs:
            if ID not in id_set:
                self.list_IDs.remove(ID)

        self._filter_IDs_for_apnea()

        self.on_epoch_end()

    def _filter_IDs_for_apnea(self, EPOCH_LEN = 30):
        """Removes IDs that are entirely apneic/hypopneic"""
        IDs_copy = self.list_IDs.copy()
        
        for ID in IDs_copy:
            with self.data_path.joinpath(ID + '.p').open('rb') as fh:
                data = pickle.load(fh)

            lo_orig, hi_orig = data["staging"][-1][:-1]
            start_epoch = lo_orig//30
            end_epoch = hi_orig//30

            apnea_free_epochs = []

            for epoch in range(start_epoch, end_epoch + 1):
                for epoch in self.apnea_dict[ID]:
                    apnea_free_epoch = True
                    if self.apnea_dict[ID][epoch] == 'A/H':
                        apnea_free_epoch = False
                apnea_free_epochs.append(apnea_free_epoch)

            # Remove ID if all epochs are apnea/hypopnea
            if not any(apnea_free_epochs):
                self.list_IDs.remove(ID)
                
    @staticmethod
    def _featurize(x, channel):
        """Use Analysis class to convert raw signal to envelope at 10hz for a specified channel"""
        analyzer = Analyzer(100)
        emg_flag = True if channel in ["Chin", "R Leg", "L Leg"] else False
        return analyzer.analyze(x, emg_flag)


    def _create_label(self, center, events):
        """
        Function to check if a given center point falls within an eps neighborhood of an RSWA event
        """
        y = np.array([1., 0., 0.])
        if events:
            for event in events:
                if y[1:].sum() > 0:
                    break

                event_start, event_end, event_type = event
                event_types = [None, 'P', 'T']
                dim = event_types.index(event_type)
                e = self.eps * self.DOWNSAMPLED_RATE
                val = 0.
                # When the center point is within 0.1s of the labeled event
                # we consider it strictly positive
                if event_start - 1 <= center <= event_end + 1:
                    # val = 1
                    if event_end - event_start == self.DOWNSAMPLED_RATE:
                        val = max(1. - (round(center - event_start, 0) / 10), 0)
                    else:
                        val = 1.

                # If the center is in an eps neighborhood before or after the labeled event
                # it is assigned a probability which is linear
                elif event_start - e < center < event_start:
                    val = 1. - (round(2 * (event_start - center), 0) / 10)

                elif event_end < center < event_end + e:
                    val = 1. - (round(2 * (center - event_end), 0) / 10)

                y[dim] = val
                y[0] = 1. - val
        return y
    
    @staticmethod
    def _window_contains_apnea(apneas: list, center: int):
        """ Returns True if center falls inside apneic epoch and False otherwise
        """
        for (e_start, e_end) in apneas:
            if e_start <= center <= e_end:
                return True
        return False


    @staticmethod
    def _time_to_indices(event_list: list, start: int, end: int, downsampled_rate: int = 10):
        """Converts times to indices. Also clip events if they extend past start and
        end arguments. Returns a list with elements that are between the start and
        end arguments."""
        new_list = []
        assert start < end, f'End is {end} but start is {start}!'
        for (e_start, e_end, e_type) in event_list:
            #print(f'tuple: {(e_start, e_end, e_type)}')
            if (start < e_end < end) or (e_start < end < e_end):
                e_start = max(e_start, start)
                e_end = min(e_end, end)
                e_start -= start
                e_end -= start
                new_list.append((round(e_start * downsampled_rate),
                         round(e_end * downsampled_rate), e_type))
        return new_list

    @staticmethod
    def _check_labels(y):
        for i, v in enumerate(y):
            if v.sum() != 1:
                print(i, v)
        assert np.allclose(y.sum(1), np.ones(len(y))), f"labels do not all sum to 1"


    def count_events(self, event_type=None):
        counts = {'T':0, 'P':0}
        for ID in self.list_IDs:
            with self.data_path.joinpath(ID + ".p").open("rb") as f_in:
                data = pickle.load(f_in)
            for event in data["rswa_events"]:
                if event[-1] not in counts:
                    continue
                counts[event[-1]] += 1
        return counts


    def __len__(self):
        """Number of batches per epoch"""
        return len(self.list_IDs)


    def __getitem__(self, index):
        """Generate one batch of data which is a single REM subsequence in this case"""
        ID = self.list_IDs[index]
        return self.__data_generation(ID)
    
    def __getitem_for_ID__(self, ID):
        """Generate one batch of data which is a single REM subsequence in this case"""
        assert ID in self.list_IDs, f'{ID} not in list_IDs!'
        
        return self.__data_generation(ID)


    def __data_generation(self, ID):
        """Generate one batch of samples where X shape is (n_samples, *dim, n_channels)"""
        X, y = self._sample(ID)
        # self._check_labels(y)
        return X, y


    def _sample(self, ID):
        """Return ALL windows w.r.t. a given REM subsequence and window size"""

        # read in data w.r.t. specified ID and epoch range
        with self.data_path.joinpath(ID + ".p").open("rb") as f_in:
            data = pickle.load(f_in)

        self.DOWNSAMPLED_RATE = 10
        events = data["rswa_events"]
        signals = data["signals"]
        
        # get apneas in units of epochs
        apnea_epochs = [epoch for epoch in self.apnea_dict[ID] if self.apnea_dict[ID][epoch] == 'A/H']
        
        # convert apneas to units of indices
        apneas = [((epoch-1)*30*self.DOWNSAMPLED_RATE, \
                   epoch*30*self.DOWNSAMPLED_RATE) for epoch in apnea_epochs]

        lo_orig, hi_orig = data["staging"][-1][:-1]
        events = self._time_to_indices(event_list=events, start=lo_orig, end=hi_orig,
                                       downsampled_rate = self.DOWNSAMPLED_RATE)

        signals = {c:self._featurize(signals[c][-(hi_orig - lo_orig):].ravel(), c) for c in self.channel_list}

        lo = 0
        hi = len(signals["Chin"])
        window = self.window_size * self.DOWNSAMPLED_RATE
        #offset = 5 * self.DOWNSAMPLED_RATE
        offset = 0

        # Calculate length
        length = hi
        for (e_start, e_end, e_type) in apneas:
            length -= (e_start - e_end + 1)

        X = np.zeros((length, *self.dim), dtype=np.float32)
        y = np.zeros((length, self.n_classes))
        # labels = np.empty((hi, self.n_channels), dtype=np.float32)
        counter = 0

        for center in range(offset, hi - offset):
            start = int(center - window // 2)
            stop = int(center + window // 2)

            if self._window_contains_apnea(apneas=apneas, center=center, window=window):
                continue

            # need to add 1 position (second) to start or stop if window size is odd (choosing start for now)
            if self.window_size % 2 != 0:
                start -= self.DOWNSAMPLED_RATE

            if start < lo:
                # left padding needed
                padding = np.zeros(abs(start))
                cut = window - len(padding)
                out = np.array([np.hstack((padding, signals[c][:cut])) for c in signals]).T
            elif stop > hi:
                # right padding needed
                padding = np.zeros(stop - hi)
                cut = window - len(padding)
                out = np.array([np.hstack((signals[c][-cut:], padding)) for c in signals]).T
            else:
                out = np.array([signals[c][start:stop] for c in signals]).T

            if self.data_format == "img":
                out = self._format_as_img(out)

            X[counter] = out

            y[counter] = self._create_label(center, events)

            counter += 1

        return X[offset:-offset], y[offset:-offset]

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import nntrader.marketdata.globaldatamatrix as gdm
import numpy as np
import logging
from nntrader.tools.configprocess import parse_time
from nntrader.tools.data import pricenorm2d, pricenorm3d, get_volume_forward
import time
import nntrader.marketdata.replaybuffer as rb
from tempfile import mkdtemp
import os.path as path

MIN_NUM_PERIOD = 3

class DataMatrices():
    def __init__(self, start, end, access_period,
                 trade_period, global_period, with_replay_buffer=True, batch_size=50,
                 fake_ratio=0.999, volume_average_days=30, buffur_bias_ratio=0,
                 save_memory_mode=False, norm_method="absolute", fake_data=False,
                 coin_filter=1, window_size=50, train_portion=0.7, is_permed=True, feature_number=1,
                 validation_portion=0.15, test_portion=0.15, validation_reversed=True, portion_reversed=False, online=False):
        """
        :param start: Unix time
        :param end: Unix time
        :param access_period: the data access period of the input matrix.
        :param trade_period: the trading period of the agent.
        :param global_period: the data access period of the global price matrix.
                              if it is not equal to the access period, there will be inserted observations
        :param coin_filter: number of coins that would be selected
        :param window_size: periods of input data
        :param train_portion: portion of training set
        :param is_permed: if False, the sample inside a mini-batch is in order
        :param validation_portion: portion of cross-validation set
        :param test_portion: portfion of test set
        :param portion_reversed: if False, the order to sets are [train, validation, test]
        else the order is [test, validation, train]
        """
        # assert window_size >= MIN_NUM_PERIOD
        self.__price_related_features = [0]
        if feature_number==1:
            type_list=["close"]
        elif feature_number==2:
            type_list=["close","volume"]
        elif feature_number==3:
            type_list=["close","high","low"]
            self.__price_related_features.append(1)
            self.__price_related_features.append(2)
        else:
            raise ValueError("feature number could not be %s" % feature_number)
        self.__coin_no = coin_filter
        self.__norm_method = norm_method
        self.__save_memory_mode = save_memory_mode
        self.__features = type_list
        self.feature_number=feature_number
        self._access_skip_distance = int(access_period / global_period)
        self._trade_skip_distance = int(trade_period / global_period)
        volume_forward = get_volume_forward(end-start,
                                            test_portion+validation_portion, portion_reversed)
        self.__history_manager = gdm.HistoryManager(coin_number=coin_filter, end=end,
                                                    volume_average_days=volume_average_days,
                                                    volume_forward=volume_forward, online=online)
        if not fake_data:
            self.__global_data = self.__history_manager.get_global_data_matrix(start,
                                                                               end,
                                                                               period=global_period,
                                                                               features=type_list)
        else:
            self.__global_data = gdm.FakeHistoryManager(coin_filter).get_global_data_matrix(start,
                                                                                            end,
                                                                                            global_period)
        self.__global_weights = (np.ones(self.__global_data.shape[1:]) / self.__coin_no).transpose()
        for i in self.__price_related_features:
            self.__global_data[i] = self.__price_norm(self.__global_data[i])

        if self.feature_number > 1:
            self.__global_data[1:] = np.roll(self.__global_data[1:], self._access_skip_distance-1, axis=2)
            self.__global_data[1:,:,:self._access_skip_distance-1] = 0
        self._window_size = window_size
        self.__removeLastNaNs()
        self.__divide_data(train_portion,
                           validation_portion,
                           test_portion,
                           portion_reversed,
                           validation_reversed)
        self._portion_reversed = portion_reversed
        self.__fake_ratio = fake_ratio
        self.__is_permed = is_permed
        self.__make_decay_prices()
        self._index_in_epoch = self._train_ind[0]
        self._completed_epochs = 0
        self.__batch_size = batch_size
        self.__replay_buffer = None
        if with_replay_buffer:
            end_index = self._train_ind[-1]
            self.__replay_buffer = rb.ReplayBuffer(start_index=self._train_ind[0],
                                                   end_index=end_index,
                                                   sample_bias=buffur_bias_ratio,
                                                   batch_size=self.__batch_size,
                                                   coin_number=self.__coin_no,
                                                   is_permed=self.__is_permed)
        else:
            self.__permutation()

        if not save_memory_mode:
            self.__processed_matrices = np.empty([self.__global_data.shape[2]-self._window_size,
                                                  feature_number,
                                                  self.__coin_no,
                                                  self._window_size+1])
        else:
            # in the save memory mode, processed data would be saved on the disk
            filename = path.join(mkdtemp(), "processed_matrices.dat")
            self.__processed_matrices = np.memmap(filename, dtype='float32', mode='w+',
                                                  shape=(self.__global_data.shape[2] -
                                                         self._window_size,
                                                         feature_number,
                                                         self.__coin_no,
                                                         self._window_size+1))
        self._sample_count = np.zeros((self.__global_data.shape[2],), dtype=np.int32)

        logging.info("the number of training examples is %s, of cross_validation examples is %s"
                     ", of test examples is %s" % (self.num_train_samples, self.num_validation_samples, self.num_test_samples))
        logging.debug("the training set is from %s to %s" % (min(self._train_ind), max(self._train_ind)))
        logging.debug("the cross_validation set is from %s to %s" % (min(self._val_ind), max(self._val_ind)))
        logging.debug("the test set is from %s to %s" % (min(self._test_ind), max(self._test_ind)))

    @property
    def global_weights(self):
        return self.__global_weights

    def __price_norm(self, m):
        if self.__norm_method == "absolute":
            return m
        elif self.__norm_method == "relative":
            pricenorm2d(m, None, norm_method="relative")
            return m
        else:
            raise ValueError("there is no norm method called %s" % self.__norm_method)

    @staticmethod
    def create_from_config(config):
        """main method to create the DataMatrices in this project
        @:param config: config dictionary
        @:return: a DataMatrices object
        """
        config = config.copy()
        input_config = config["input"]
        train_config = config["training"]
        start = parse_time(input_config["start_date"])
        end = parse_time(input_config["end_date"])
        return DataMatrices(start=start,
                            end=end,
                            save_memory_mode=input_config["save_memory_mode"],
                            norm_method=input_config["norm_method"],
                            fake_ratio=input_config["fake_ratio"],
                            feature_number=input_config["feature_number"],
                            window_size=input_config["window_size"],
                            access_period=input_config["access_period"],
                            trade_period=input_config["trade_period"],
                            online=input_config["online"],
                            global_period=input_config["global_period"],
                            coin_filter=input_config["coin_number"],
                            fake_data=input_config["fake_data"],
                            is_permed=input_config["is_permed"],
                            buffur_bias_ratio=train_config["buffer_biased"],
                            batch_size=train_config["batch_size"],
                            volume_average_days=input_config["volume_average_days"],
                            validation_portion=input_config["validation_portion"],
                            test_portion=input_config["test_portion"],
                            portion_reversed=input_config["portion_reversed"],
                            validation_reversed=input_config["validation_reversed"]
                            )

    @property
    def global_matrix(self):
        return self.__global_data

    @property
    def coin_list(self):
        return self.__history_manager.coins

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def completed_epochs(self):
        return self._completed_epochs

    @property
    def num_train_samples(self):
        return self._num_train_samples

    @property
    def num_validation_samples(self):
        return self._num_validation_samples

    @property
    def skiped_validation_indices(self):
        return self._val_ind[:-(self._window_size*self._access_skip_distance+self._trade_skip_distance):self._trade_skip_distance]

    @property
    def skiped_test_indices(self):
        return self._test_ind[:-(self._window_size*self._access_skip_distance+self._trade_skip_distance):self._trade_skip_distance]

    @property
    def num_test_samples(self):
        return self._num_test_samples

    @property
    def sample_count(self):
        return self._sample_count

    def append_experience(self, online_sample=None):
        """
        :param online_sample: a dictionary of {"M":(coin,windowsize+1) array, "w":(coin,)array}
        "M" is a combine of X and y.
        Let it be None if in the backtest case.
        """
        if not self._portion_reversed:
            # backtest case
            for i in range(self._trade_skip_distance):
                self._train_ind.append(self._train_ind[-1]+1)
                appended_index = self._train_ind[-1]
                self.__replay_buffer.append_experience(appended_index)
        elif online_sample:
            # realtime trading case
            self.__global_weights = np.concatenate((self.__global_weights,
                                                    online_sample["w"][None, 1:]),
                                                   axis=0)
            self.__processed_matrices = np.concatenate((self.__processed_matrices,
                                                        online_sample["M"][None, :]),
                                                       axis=0)
            self._sample_count = np.concatenate((self._sample_count, np.ones(shape=(1,))))
            self._train_ind.append(self._train_ind[-1]+1)
            appended_index = self._train_ind[-1]
            self.__replay_buffer.append_experience(appended_index)
        else:
            msg = "you must provide a online sample"
            logging.error(msg)
            raise ValueError(msg)

    def __make_decay_prices(self):
        self._fake_prices = np.array([self.__fake_ratio**(self._window_size - i - 1)
                                     for i in range(self._window_size + 1)])

    def get_test_set(self):
        return self.__pack_samples(self.skiped_test_indices)

    def get_cross_validation_set(self):
        return self.__pack_samples(self.skiped_validation_indices)

    def get_training_set(self):
        return self.__pack_samples(self._train_ind[:-self._window_size])

    def next_batch(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """
        batch = self.__next_batch_by_replay_buffer()
        return batch

    def __next_batch_by_replay_buffer(self):
        batch = self.__pack_samples([exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)
        logging.debug("the max of indexs is {}".format(max(indexs)))
        last_w = self.__global_weights[indexs-1, :]
        w = [self.__global_weights[i, :] for i in indexs]
        M = [self.getSubMatrix(index) for index in indexs]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, -1]
        return {"X": X, "y": y, "last_w": last_w, "w": w}

    def __process_submatrix(self, ind):
        if self.feature_number == 2:
            mc_x_price = self.__global_data[[0], :, ind
            :ind + self._window_size * self._access_skip_distance
            :self._access_skip_distance]
            mc_x_volume = self.__global_data[[1], :, ind:ind + self._window_size * self._access_skip_distance]
            # fill nan volume with 0
            mc_x_volume = np.nan_to_num(mc_x_volume)
            # add volume up
            mc_x_volume = np.sum(mc_x_volume.reshape(1, self.__coin_no, self._window_size, self._access_skip_distance),
                                 axis=3)
            mc_x = np.concatenate([mc_x_price, mc_x_volume], axis=0)

            mc_y_price = self.__global_data[0, :,
                         ind + (self._window_size-1) * self._access_skip_distance + self._trade_skip_distance]
            mc_y_volume = self.__global_data[1, :, ind + self._window_size * self._access_skip_distance]
            mc_y = np.stack([mc_y_price, mc_y_volume], axis=0)[:, :, np.newaxis]

            mc = np.concatenate([mc_x, mc_y], axis=2)
        elif self.feature_number == 1:
            mc_x = self.__global_data[[0], :, ind
            :ind + self._window_size * self._access_skip_distance
            :self._access_skip_distance]
            mc_y = self.__global_data[0, :,
                   ind + (self._window_size-1)*self._access_skip_distance + self._trade_skip_distance][np.newaxis, :,
                   np.newaxis]
            mc = np.concatenate([mc_x, mc_y], axis=2)
            # mc = np.concatenate((np.ones((1, 1, self._window_size + 1)), mc), axis=1)
        elif self.feature_number == 3:
            mc = np.empty((self.feature_number, self.__coin_no, self._window_size+1))
            for i in range(3):
                if i == 0:
                    # close
                    mc_x = self.__global_data[[i], :, ind
                    :ind + self._window_size * self._access_skip_distance
                    :self._access_skip_distance]
                if i == 1:
                    # high
                    mc_x = self.__global_data[[i], :, ind:ind + self._window_size * self._access_skip_distance]
                    mc_x = np.max(
                        mc_x.reshape(1, self.__coin_no, self._window_size, self._access_skip_distance),
                        axis=3)
                if i == 2:
                    # low
                    mc_x = self.__global_data[[i], :, ind:ind + self._window_size * self._access_skip_distance]
                    mc_x = np.min(
                        mc_x.reshape(1, self.__coin_no, self._window_size, self._access_skip_distance),
                        axis=3)
                mc_y = self.__global_data[i, :,
                       ind + (self._window_size-1)*self._access_skip_distance + self._trade_skip_distance][np.newaxis, :,
                    np.newaxis]
                mc[i] = np.concatenate([mc_x, mc_y], axis=2)
        else:
            raise ValueError()

        if self.__norm_method == "absolute":
            m = pricenorm3d(mc, self.__features, self.__norm_method, self.__fake_ratio)
        else:
            m = mc
        return m

    # volume in y is the volume in next access period
    def getSubMatrix(self, ind):
        self._sample_count[ind] += 1
        # use list of index to preserve the dimension
        # price is the last price, therefore the starting index is the
        if self._sample_count[ind] == 1:
            self.__processed_matrices[ind] = self.__process_submatrix(ind)
        return self.__processed_matrices[ind]

    def __permutation(self):
        self._perm = np.array(self._train_ind[:-self._window_size*self._access_skip_distance]).copy()
        np.random.shuffle(self._perm)

    def __price_normalization(self, m, i):
        row = m[i]
        m[i] = row / row[-2]

    def __removeLastNaNs(self):
        i = -1
        while(np.isnan(self.__global_data[0, :, i]).any()):
            i -= 1
        i += 1
        self._num_periods = self.__global_data.shape[2] + i

    def __divide_data(self, train_portion, validation_portion, test_portion, portion_reversed, validation_reversed):
        s = float(train_portion + validation_portion + test_portion)
        if portion_reversed:
            portions = np.array([test_portion, test_portion + validation_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._test_ind, self._val_ind, self._train_ind = np.split(indices, portion_split)
        elif validation_reversed:
            portions = np.array([validation_portion, train_portion + validation_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._val_ind, self._train_ind, self._test_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion, train_portion + validation_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._train_ind, self._val_ind, self._test_ind = np.split(indices, portion_split)

        self._train_ind = self._train_ind[:-(self._window_size*self._access_skip_distance +
                                             self._trade_skip_distance)]
        # NOTE(zhengyao): change the logic here in order to fit both
        # reversed and normal version
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_validation_samples = len(self.skiped_validation_indices)
        self._num_test_samples = len(self.skiped_test_indices)

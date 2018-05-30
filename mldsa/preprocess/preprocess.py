import os
import pandas as pd
import numpy as np
import pathlib


class Preprocess:
    def __init__(self, data_dir, course_name, names, timestamps):
        if not data_dir.endswith('/'):
            data_dir += '/'
        if not os.path.isdir(data_dir):
            raise NotADirectoryError("{}: indicated data directory does not exist.".format(self.__class__.__name__))
        if not os.path.isdir(data_dir + 'csv/'):
            raise NotADirectoryError("{}: indicated data directory does not "
                                     "have a csv sub-folder.".format(self.__class__.__name__))
        self.data_dir = data_dir
        if not os.path.isfile(data_dir + 'csv/' + course_name + '.csv'):
            raise FileNotFoundError("{}: indicated csv file does not exist.".format(self.__class__.__name__))
        self.course_name = course_name
        if not isinstance(names, list) and all([isinstance(n, str) for n in names]):
            raise ValueError("{}: names should be a list of strings.".format(self.__class__.__name__))
        if not isinstance(timestamps, list) and all([isinstance(t, str) for t in timestamps]):
            raise ValueError("{}: timestamps should be a list of strings.".format(self.__class__.__name__))
        self.names = names
        self.timestamps = timestamps
        self.table = pd.read_csv(data_dir + 'csv' + course_name + '.csv',
                                 header=None, names=self.names, index_col='observed_event_id',
                                 parse_dates=self.timestamps, date_parser=pd.core.tools.datetimes.to_datetime)
        self.table.reset_index(inplace=True)
        self.feature = None
        self.label = None
        self.x = None
        self.y = None
        print("{}: Data loaded:".format(self.__class__.__name__))
        print(' - number of records: {} - column names: {}'.format(self.table.shape[0], self.table.columns.values.tolist()))

    def process(self, *args):
        pass


class RawEventPreprocess(Preprocess):
    def __init__(self, data_dir, course_name):
        super(RawEventPreprocess, self).__init__(data_dir, course_name,
                                                 ['observed_event_id', 'user_id',
                                                  'observed_event_timestamp', 'observed_event_type'],
                                                 ['observed_event_timestamp'])

    def process(self, start_timestamp, duration, time_granularity):
        # Check
        if not (isinstance(duration, int) and duration > 0):
            raise ValueError("{}: duration should be a positive integer.".format(self.__class__.__name__))
        if not (isinstance(time_granularity, int) and time_granularity > 0):
            raise ValueError("{}: time granularity should be a positive integer.".format(self.__class__.__name__))
        if not duration % time_granularity == 0:
            raise ValueError("{}: duration should be a multiple of time granularity.".format(self.__class__.__name__))
        # Convert event_type to event_id and renumbering user_id
        event_types = self.table.observed_event_type.unique()
        event_id_dict = {x: i for i, x in enumerate(event_types)}
        self.table['observed_event_type'] = self.table['observed_event_type'].apply_layer(lambda x: event_id_dict[x])
        self.table.rename(columns={'observed_event_type': 'event_id'}, inplace=True)
        user_ids = self.table.user_id.unique()
        user_id_dict = {x: i for i, x in enumerate(user_ids)}
        self.table['user_id'] = self.table['user_id'].apply_layer(lambda x: user_id_dict[x])
        # Calculate time diffs
        self.table['observed_event_timestamp'] = self.table['observed_event_timestamp'].apply_layer(lambda x: (x - start_timestamp).days)
        self.table.rename(columns={'observed_event_timestamp': 'date'}, inplace=True)
        self.table = self.table[(self.table['date'] >= 0) & (self.table['date'] < duration)]
        # Generate x matrix
        self.feature = self.table.groupby(['user_id', 'date', 'event_id'], as_index=False)['observed_event_id'].agg(['count'])
        self.feature.reset_index(inplace=True)
        self.x = np.zeros(self.feature.iloc[:, :-1].max() + 1)
        np.add.at(self.x, self.feature.iloc[:, :-1].T.values.tolist(), self.feature['count'])
        # Generate y matrix
        self.label = self.table.groupby(['user_id'], as_index=False)['date'].agg(['max'])
        self.label.reset_index(inplace=True)
        self.label['max'] = self.label['max'].apply_layer(lambda x: x + 1)
        self.label = self.label[self.label['max'] < duration]
        self.y = np.zeros(self.label.max() + 1)
        np.add.at(self.y, self.label.T.values.tolist(), 1)
        self.y = np.cumsum(self.y, axis=1)
        self.y = np.ones_like(self.y) - self.y
        # Further aggregations
        self.x = np.sum(self.x.reshape(self.x.shape[0], self.x.shape[1] // time_granularity,
                                       time_granularity, self.x.shape[2]), axis=2)
        self.y = np.sum(self.y.reshape(self.y.shape[0], self.y.shape[1] // time_granularity,
                                       time_granularity), axis=2)
        self.y[self.y > 0] = 1
        # Rescale features
        self.x = self.x / (np.max(self.x, axis=0) + 1e-14)
        pathlib.Path(self.data_dir + self.course_name).mkdir(parents=True, exist_ok=True)
        np.save(self.data_dir + self.course_name + '/x.npy', x)
        np.save(self.data_dir + self.course_name + '/y.npy', y)
        np.save(self.data_dir + self.course_name + '/event_types.npy', event_types)
        np.save(self.data_dir + self.course_name + '/user_ids.npy', user_ids)
        print("{}: Process finished.".format(self.__class__.__name__))


import time

VERBOSITY_OPTIONS = {
    0: 'nothing',
    1: 'mission progress bar',
    2: 'mission itemize, epoch progress bar',
    3: 'mission itemize, epoch itemize'
}


class Logger:
    def __init__(self, paras):
        self.verbosity = paras.log.verbosity
        self.mission_tracker = None
        self.epoch_tracker = None
        print("{}: selected verbosity {}.".format(self.__class__.__name__, self.verbosity))

    def start_mission(self, paras):
        from functools import reduce
        from operator import mul
        it_lists = paras.get_iter_lists()
        total_mission = reduce(mul, [len(v) for v in it_lists.values()], 1)
        self.mission_tracker = Tracker(total_mission)
        if self.verbosity > 0:
            print('###### Mission Statistics: '
                  'Total#:{} ({})'.format(total_mission,
                                          ', '.join(['{}#:{}'.format(k, len(v)) for k, v in it_lists.items()])))
        self.mission_tracker.start()

    def log_mission(self, paras):
        self.mission_tracker.tick()
        if self.verbosity == 1:
            self.mission_tracker.print_scroll_back('### Mission:{} ({})'.format(
                self.mission_tracker.print_progress_bar(length=60),
                self.mission_tracker.print_time_estimation()))
        if self.verbosity > 1:
            print('### Mission:{} ({}) '.format(
                self.mission_tracker.get_cur(),
                self.mission_tracker.print_time_estimation()) +
                  ', '.join(['{}:{}'.format(k, v) for k, v in paras.get_cur_iter_paras().items()]))
        self.mission_tracker.update()

    def start_epoch(self, paras):
        self.epoch_tracker = Tracker(paras.train.nepochs)
        self.epoch_tracker.start()

    def log_epoch(self, paras, result):
        self.epoch_tracker.tick()
        if self.verbosity == 2:
            self.epoch_tracker.print_scroll_back(
                ' - epoch:{}'.format(self.epoch_tracker.print_progress_bar(length=48)) +
                ' - loss_train:{:05.4e} - loss_test:{:05.4e} - '.format(*result.get_singe_loss()) +
                ' - '.join(['{0}_train:{1:05.4e} - {0}_test:{2:05.4e}'.format(n, *m)
                            for n, m in result.get_singe_metric().items()]))
        if self.verbosity == 3:
            print(' - epoch:{}'.format(self.epoch_tracker.get_cur()) +
                  ' - loss_train:{:05.4e} - loss_test:{:05.4e} - '.format(*result.get_singe_loss()) +
                  ' - '.join(['{0}_train:{1:05.4e} - {0}_test:{2:05.4e}'.format(n, *m)
                              for n, m in result.get_singe_metric().items()]))
        self.epoch_tracker.update()


class Tracker:
    def __init__(self, total):
        assert isinstance(total, int) and total > 0, \
            "{}: total {} should be a positive integer.".format(self.__class__.__name__, total)
        self.total = total
        self.cur = None
        self.start_time = None
        self.cur_time = None
        self.prev_time = None

    @staticmethod
    def format_time(delta_time):
        delta_time = int(delta_time)
        assert delta_time >= 0, "Tracker: negative time interval {} encountered.".format(delta_time)
        if delta_time < 60:
            return '{}s'.format(delta_time)
        elif delta_time < 3600:
            return '{}m{}s'.format(delta_time//60, delta_time % 60)
        elif delta_time < 3600*24:
            return '{}h{}m{}s'.format(delta_time//3600, (delta_time % 3600)//60, delta_time//60)
        else:
            return '{}d{}h{}m{}s'.format(delta_time//3600//24, (delta_time % (3600*24))//3600,
                                         (delta_time % 3600)//60, delta_time//60)

    @staticmethod
    def format_progress_bar(iteration, total, length, prefix='', suffix=''):
        filled_length = int(length * iteration // total)
        bar = '=' * (filled_length - int(iteration == total)) + '>' + '-' * (length - filled_length - 1)
        return '{0} [{1}] {2:05.1f}% {3}'.format(prefix, bar, 100 * (iteration / float(total)), suffix)

    def get_time_estimation(self):
        time_used = self.cur_time - self.prev_time
        time_past = self.cur_time - self.start_time
        time_est = time_past / self.cur * (self.total - self.cur)
        return time_used, time_past, time_est

    def print_time_estimation(self):
        return 'spt:{}-pst:{}-est:{}'.format(*[Tracker.format_time(t)
                                                   for t in self.get_time_estimation()])

    def print_progress_bar(self, length):
        return Tracker.format_progress_bar(self.cur, self.total, length)

    def print_scroll_back(self, string):
        print(string, end='\r' if self.cur < self.total else '\n')

    def start(self):
        self.cur = 0
        self.start_time = time.time()
        self.prev_time = self.cur_time = self.start_time

    def tick(self):
        self.cur += 1
        self.cur_time = time.time()

    def update(self):
        self.prev_time = self.cur_time
        if self.cur == self.total:
            self.end()

    def end(self):
        self.cur = None
        self.start_time = None
        self.prev_time = self.cur_time = None

    def get_cur(self):
        return self.cur

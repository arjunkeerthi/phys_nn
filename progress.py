from threading import Thread

class ProgressBarThread(Thread):
    def __init__(self, total, length=50, prefix="", suffix=""):
        super(ProgressBarThread, self).__init__()
        self.progress = None
        self.total = total
        self.length = length
        self.prefix = prefix
        self.suffix = suffix
        self.stopped = False
        self.finished = False

    def update(self, progress):
        self.progress = progress

    def run(self):
        while not self.stopped:
            if self.progress is not None:
                percentage = int(self.progress/self.total*100)
                fill_length = int(self.length*self.progress/self.total)
                empty_length = self.length - fill_length
                bar = '#'*fill_length + ' '*empty_length
                print(f"\r{self.prefix}[{bar}] {percentage}%{self.suffix}", end="", flush=True)
        if self.finished and self.total - self.progress <= 1:
            bar = '#'*self.length
            print(f"\r{self.prefix}[{bar}] {100}%{self.suffix}", end="", flush=True)

    def stop(self, finished=False):
        self.stopped = True
        self.finished = True

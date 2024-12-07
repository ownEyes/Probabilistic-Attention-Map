import logging


class Log():

    def __init__(self, save_dir):
        self.logger = logging.getLogger()
        self.logger.setLevel(level=logging.DEBUG)
        self.f1 = logging.Formatter(
            fmt='[time]%(asctime)s %(message)s \n')
        self.f2 = logging.Formatter(
            fmt='[time]%(asctime)s %(message)s \n')
        self.save_dir = save_dir

    def add_StreamHandler(self):

        self.hand = logging.StreamHandler()  # console
        self.hand.setLevel(level=logging.DEBUG)
        # self.hand.setFormatter(self.f1)
        self.logger.addHandler(self.hand)

    def add_FileHandler(self):

        self.filehand = logging.FileHandler(filename='{}'.format(
            self.save_dir+'log.txt'), encoding='utf-8')
        self.filehand.setLevel(level=logging.DEBUG)
        # self.filehand.setFormatter(self.f2)
        self.logger.addHandler(self.filehand)

    def run(self):
        self.add_StreamHandler()
        self.add_FileHandler()
        return self.logger


# if __name__ == '__main__':
#     import pprint
#     import yaml
#     with open('../configs/experiment1.yaml', 'r') as fin:
#         cfg = yaml.safe_load(fin)
#     dict_cfg = pprint.pformat(cfg)
#     log = Log('../logs/').run()
#     # log.info(f"{dict_cfg}")
#     log.info('{}'.format(dict_cfg))
#     log.info('hello world')

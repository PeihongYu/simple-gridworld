import tensorboardX
import numpy as np
import utils
np.set_printoptions(precision=2)


class tb_writer:
    def __init__(self, model_dir, agent_num, use_prior):
        self.model_dir = model_dir
        self.tb_writer = tensorboardX.SummaryWriter(model_dir)
        self.csv_file, self.csv_logger = utils.get_csv_logger(self.model_dir, mode="a")
        self.agent_num = agent_num
        self.use_prior = use_prior
        self.pweight = 0
        self.now_len = 0
        self.max_len = 100
        self.is_full = False
        self.ep_num = 0
        self.frames_num = 0
        self.frames = np.zeros(self.max_len)
        self.returns = np.zeros([self.max_len, agent_num * (1 + use_prior)])

    def update_csv_logger(self, status):
        if status["episode"] == 0:
            self.csv_file, self.csv_logger = utils.get_csv_logger(self.model_dir, mode="w")
            header = ""
            self.csv_logger.writerow(header)
        else:
            self.csv_file.flush()

    def add_info(self, frames, returns, pweight=0):
        self.frames[self.now_len] = frames
        self.returns[self.now_len] = returns
        self.pweight = pweight
        self.now_len += 1
        self.ep_num += 1
        self.frames_num += frames
        # self.csv_logger.

        if self.now_len >= self.max_len:
            self.now_len = 0
            self.is_full = True

    def add_info_batch(self, frames, returns):
        sid = self.now_len
        l = min(frames.size, self.max_len - sid)

        self.frames[sid:sid + l] = frames[:l]
        self.returns[sid:sid + l] = returns[:l, ]
        self.now_len += l

        if self.now_len >= self.max_len:
            self.now_len = 0
            self.is_full = True
        if l < frames.size:
            l = frames.size - l
            self.frames[:l] = frames[l:]
            self.returns[:l] = returns[l:]
            self.now_len = l
            self.is_full = True

    def log(self, idx):
        if self.is_full:
            mean_frames = self.frames.mean()
            mean_returns = self.returns.mean(axis=0)
        else:
            mean_frames = self.frames[:self.now_len].mean()
            mean_returns = self.returns[:self.now_len].mean(axis=0)

        print("episode ", self.ep_num, " average frames: ", mean_frames, " average returns: ", mean_returns)

        self.tb_writer.add_scalar("frames", mean_frames, idx)
        self.tb_writer.add_scalar("ep_frames", mean_frames, self.ep_num)
        for aid in range(self.agent_num):
            self.tb_writer.add_scalar("returns_a" + str(aid), mean_returns[aid], idx)
            self.tb_writer.add_scalar("ep_returns_a" + str(aid), mean_returns[aid], self.ep_num)
            if self.use_prior:
                self.tb_writer.add_scalar("shadow_returns_a" + str(aid), mean_returns[self.agent_num + aid], idx)
                self.tb_writer.add_scalar("ep_shadow_returns_a" + str(aid), mean_returns[self.agent_num + aid],
                                          self.ep_num)
        self.tb_writer.add_scalar("pweight", self.pweight, idx)
        self.tb_writer.add_scalar("ep_pweight", self.pweight, self.ep_num)

        return mean_returns[:self.agent_num]

import numpy as np
import torch

dtype = torch.float

class Replay():
    def __init__(self, sample_size, max_size, s_shape):
        self.sample_size = sample_size
        self.max_size = max_size
        
        if type(s_shape) == int:  # if integer is passed in
            self.S = torch.zeros((max_size, s_shape), dtype = dtype)
            self.Sp = torch.zeros((max_size, s_shape), dtype = dtype)
        else:
            self.S = torch.zeros((max_size, *s_shape), dtype = dtype)
            self.Sp = torch.zeros((max_size, *s_shape), dtype = dtype)
        self.D = torch.zeros(max_size, dtype = dtype)
        self.R = torch.zeros(max_size, dtype = dtype)
        self.A = torch.zeros(max_size, dtype = torch.long)
        self.buffers = [[None] * max_size for _ in range(5)]
        self.curr_size = 0
        self.ix_to_replace = 0

    def push(self, s, a, r, sp, done):
        if self.curr_size < self.max_size:
            self.curr_size += 1
        self.S[self.ix_to_replace, :] = s
        self.Sp[self.ix_to_replace, :] = sp 
        self.D[self.ix_to_replace] = float(done)
        self.R[self.ix_to_replace] = float(r) 
        self.A[self.ix_to_replace] = a
        self.ix_to_replace = (self.ix_to_replace + 1) % self.max_size

    def sample(self):
        ixs = np.random.randint(self.curr_size, size = self.sample_size)
        return self.S[ixs, :], self.A[ixs], self.R[ixs], self.Sp[ixs, :], self.D[ixs]

    def save(self, dir):
        # save the data in some directory
        torch.save(self.S, dir + "_S.pt")
        torch.save(self.Sp, dir + "_Sp.pt")
        torch.save(self.A, dir + "_A.pt")
        torch.save(self.R, dir + "_R.pt")
        torch.save(self.D, dir + "_D.pt")


    def load(self, dir):
        self.S = torch.load(dir + "_S.pt")
        self.Sp = torch.load(dir + "_Sp.pt")
        self.R = torch.load(dir + "_R.pt")
        self.A = torch.load(dir + "_A.pt")
        self.D = torch.load(dir + "_D.pt")
        self.curr_size = self.D.numel()
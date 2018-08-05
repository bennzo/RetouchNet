import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class RetouchGenerator(nn.Module):
    def __init__(self, device):
        super(RetouchGenerator, self).__init__()

        self.device = device

        ## Define layers as described in the HDRNet architecture
        # Activation
        self.activate = nn.ReLU(inplace=True)

        # Low-level layers (S)
        self.ll_conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.ll_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.ll_conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.ll_conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)

        # Local features layers (L)
        self.lf_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lf_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        # Global features layers (G)
        self.gf_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.gf_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.gf_fc1 = nn.Linear(1024, 256)
        self.gf_fc2 = nn.Linear(256, 128)
        self.gf_fc3 = nn.Linear(128, 64)

        # Linear prediction (Pointwise layer)
        self.pred_conv = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=1, stride=1)

        # Guidance map auxilary parameters
        self.pw_mat = nn.Parameter(torch.eye(3) + torch.randn(1)*1e-4, requires_grad=True)
        self.pw_bias = nn.Parameter(torch.eye(1), requires_grad=True)
        self.pw_bias_tag = nn.Parameter(torch.zeros(3,1), requires_grad=True)

        self.rho_a = nn.Parameter(torch.ones(16,3), requires_grad=True)
        self.rho_t = nn.Parameter(torch.rand(16,3), requires_grad=True)


    def forward(self, high_res, low_res):
        bg = self.create_bilateral(low_res)
        guide = self.create_guide(high_res)
        output = self.slice_and_assemble(bg, guide, high_res)
        return output

    def create_bilateral(self, low_res):
        ## TODO: Add batch normalization
        # Low-level
        ll = self.activate(self.ll_conv1(low_res))
        ll = self.activate(self.ll_conv2(ll))
        ll = self.activate(self.ll_conv3(ll))
        ll = self.activate(self.ll_conv4(ll))

        # Local features
        lf = self.activate(self.lf_conv1(ll))
        lf = self.lf_conv2(lf)  # No activation (normalization) before fusion

        # Global featuers
        gf = self.activate(self.gf_conv1(ll))
        gf = self.activate(self.gf_conv2(gf))
        gf = self.activate(self.gf_fc1(gf.view(-1, gf.shape[1]*gf.shape[2]*gf.shape[3])))
        gf = self.activate(self.gf_fc2(gf))
        gf = self.gf_fc3(gf)  # No activation (normalization) before fusion

        # Fusion
        fusion = self.activate(gf.view(-1, 64, 1, 1) + lf)

        # Bilateral Grid
        pred = self.pred_conv(fusion)
        bilateral_grid = pred.view(-1, 12, 16, 16, 8)  # Image features as a bilateral grid

        return bilateral_grid

    def create_guide(self, high_res):
        guide = high_res.view(high_res.shape[0], 3, -1)                             # (nbatch, nchannel, w*h)
        guide = torch.matmul(self.pw_mat, guide)
        guide = guide + self.pw_bias_tag
        guide = self.activate(guide.unsqueeze(1) - self.rho_t.view([16, 3, 1]))     # broadcasting to (nbatch, 16, nchannel, w*h)
        guide = guide.permute(0,3,1,2) * self.rho_a                                 # (nbatch, w*h, 16, nchannel)
        guide = guide.sum(3).sum(2) + self.pw_bias
        guide = guide.view(high_res.shape[0],high_res.shape[2],high_res.shape[3])   # return to original shape

        return guide

    def slice_and_assemble(self, bg, guide, high_res):
        # clip guide to [-1,1] to comply with 'grid_sample'
        guide = (guide / guide.max(2)[0].max(1)[0].unsqueeze(1).unsqueeze(2))*2 - 1
        bs, gh, gw = guide.shape

        output = torch.zeros((bs, 3, gh, gw)).to(self.device)

        # create xy meshgrid for bilateral grid slicing
        x = torch.linspace(-1, 1, gw)
        y = torch.linspace(-1, 1, gh)
        x_t = x.repeat([gh,1]).to(self.device)
        y_t = y.view(-1,1).repeat([1, gw]).to(self.device)
        xy = torch.cat([x_t.unsqueeze(2), y_t.unsqueeze(2)], dim=2)

        for b in range(0,bs):
            guide_aug = torch.cat([xy, guide[b].unsqueeze(2)], dim=2).unsqueeze(0).unsqueeze(0)         # augment meshgrid with z dimension (1,out_d,out_h,out_w,idx)
            slice = nn.functional.grid_sample(bg[b:b+1], guide_aug)                                     # slice bilateral grid

            # assemble output
            output[b,0,:,:] = slice[0,3,0] + slice[0,0,0]*high_res[b,0,:,:] + slice[0,1,0]*high_res[b,0,:,:] + slice[0,2,0]*high_res[b,0,:,:]
            output[b,1,:,:] = slice[0,7,0] + slice[0,4,0]*high_res[b,1,:,:] + slice[0,5,0]*high_res[b,1,:,:] + slice[0,6,0]*high_res[b,1,:,:]
            output[b,2,:,:] = slice[0,11,0] + slice[0,8,0]*high_res[b,2,:,:] + slice[0,9,0]*high_res[b,2,:,:] + slice[0,10,0]*high_res[b,2,:,:]

        return output











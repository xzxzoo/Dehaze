import torch.nn as nn
from loss import CR_loss
import torch
from torch.nn import functional as F

class Dehaze_loss(nn.Module):
    def __init__(self):
        super(Dehaze_loss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean')
        self.crloss = CR_loss.ContrastLoss()

    def consistancy_loss(self, dehaze_output_img, target_img, dehaze_output_img_2_depth_img, real_img_2_depth_img):

        diff_depth = torch.sub(dehaze_output_img_2_depth_img, real_img_2_depth_img)
        B, C, H, W = diff_depth.shape
        diff_depth = diff_depth.permute(0, 2, 3, 1)
        diff_depth = diff_depth.reshape(-1, C * H * W)
        epsilon = 1e-7
        diff_d_w = F.softmax(diff_depth, dim=-1) + epsilon
        diff_d_w = diff_d_w.reshape(B, H, W, C).permute(0, 3, 1, 2)
        diff_d_w1 = diff_d_w.expand(-1, 3, -1, -1)
        weighted_dehaze_output_img = torch.mul(dehaze_output_img, diff_d_w1)
        weighted_target_img = torch.mul(target_img, diff_d_w1)

        loss_consis = self.l1(weighted_dehaze_output_img, weighted_target_img)

        return loss_consis

    def consistancy_loss2(self, dehaze_output_img, target_img):

        loss_consis2 = self.l1(dehaze_output_img, target_img)

        return loss_consis2

    def deepestimate_loss(self, t_d1, t_d2, t_d3, o_d1, o_d2, o_d3):

        loss_deepes_1 = self.l1(t_d1, o_d1)
        loss_deepes_2 = self.l1(t_d2, o_d2)
        loss_deepes_3 = self.l1(t_d3, o_d3)
        loss_deepes = loss_deepes_1 + loss_deepes_2 + loss_deepes_3

        return loss_deepes

    def forward(self, dehaze_output_img, target_img, dehaze_output_img_2_depth_img, real_img_2_depth_img, t_d1, t_d2, t_d3, o_d1, o_d2, o_d3,
                source_img):

        loss_consis = self.consistancy_loss(dehaze_output_img, target_img, dehaze_output_img_2_depth_img, real_img_2_depth_img)
        loss_consis2 = self.consistancy_loss2(dehaze_output_img, target_img)
        deep_loss = self.deepestimate_loss(t_d1, t_d2, t_d3, o_d1, o_d2, o_d3)
        loss_CR = self.crloss(dehaze_output_img, target_img, source_img)
        loss_dehaze_total = loss_consis + deep_loss + loss_CR + loss_consis2

        return loss_consis, deep_loss, loss_CR, loss_dehaze_total, loss_consis2
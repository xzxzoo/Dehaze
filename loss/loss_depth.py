import torch
import torch.nn as nn
import torch.nn.functional as F
class Depth_loss(nn.Module):
    def __init__(self):
        super(Depth_loss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean')

    def depth_loss(self,dehaze_output_img, target_img, dehaze_output_img_2_depth_img, real_img_2_depth_img):
        diff_dehaze = torch.sub(dehaze_output_img, target_img)
        B, C, H, W = diff_dehaze.shape
        diff_dehaze = diff_dehaze.permute(0, 2, 3, 1)
        diff_dehaze = diff_dehaze.reshape(-1, C * H * W)
        epsilon = 1e-7
        diff_d_w = F.softmax(diff_dehaze, dim=-1) + epsilon
        diff_d_w = diff_d_w.reshape(B, H, W, C).permute(0, 3, 1, 2)
        diff_dehaze_w = torch.sum(diff_d_w, dim=1, keepdim=True)
        weighted_dehaze_output_img = torch.mul(dehaze_output_img_2_depth_img, diff_dehaze_w)
        weighted_real_img_2_depth_img = torch.mul(real_img_2_depth_img, diff_dehaze_w)
        loss_depth = self.l1(weighted_dehaze_output_img, weighted_real_img_2_depth_img)
        return loss_depth

    def depth_loss2(self, dehaze_output_img_2_depth_img, real_img_2_depth_img):
        loss_depth2 = self.l1(dehaze_output_img_2_depth_img, real_img_2_depth_img)
        return loss_depth2

    def forward(self, dehaze_output_img, target_img, dehaze_output_img_2_depth_img, real_img_2_depth_img):
        loss_depth2 = self.depth_loss2(dehaze_output_img_2_depth_img, real_img_2_depth_img)
        loss_depth1 = self.depth_loss(dehaze_output_img.detach(), target_img.detach(), dehaze_output_img_2_depth_img, real_img_2_depth_img)
        loss_total_depth = loss_depth2 + 0.1*loss_depth1
        return  loss_depth2, loss_depth1, loss_total_depth
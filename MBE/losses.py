import torch
import torch.nn as nn
import torch.nn.functional as F
#from MPRNet import Discriminator
import os
from config import Config
opt = Config('training.yml')
import utils

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

"""
class GANloss(nn.Module):
    def __init__(self, lr=0.0001, betasInit=(0.5, 0.9), model_dir=os.path.join(opt.TRAINING.SAVE_DIR, opt.MODEL.MODE, 'models')):
        super(GANloss, self).__init__()
        self.discriminator = Discriminator(3)    ## local_global sn patch gan
        self.cudaAvailable = torch.cuda.is_available()
        self.numOfGPUs = torch.cuda.device_count()
        self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betasInit)
        ######### Resume ###########


    def forward(self, input, mask, restored, gt, epoch, model_dir):
        self.discriminator.zero_grad()
        D_real = self.discriminator(gt, mask)
        D_real = D_real.mean().sum()*-1
        D_fake = self.discriminator(restored, mask)
        D_fake = D_fake.mean().sum()
        D_loss = torch.mean(F.relu(1.+D_real)) + torch.mean(F.relu(1.+D_fake))
        self.D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        self.D_optimizer.step()
        torch.save({'epoch': epoch,
                    'state_dict': self.discriminator.state_dict(),
                    'optimizer' : self.D_optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest_discriminor.pth"))
        return D_loss, D_fake
"""

class Otherloss(nn.Module):
    def __init__(self, extractor):
        super(Otherloss, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.cudaAvaliable = torch.cuda.is_available()
        self.numberOfGPUs = torch.cuda.device_count()

    def forward(self, input, mask, inter1, inter2, output, gt, mm):

        output_comp = mask * input + (1 - mask) * output
        holeLoss = 10 * self.l1((1 - mask) * output, (1 - mask) * gt)
        validAreaLoss = 2*self.l1(mask * output, mask * gt)

        ### Dice lose ###
        mask_loss = dice_loss(mm, mask)

        ### Reconstruction loss ###
        masks_a = mask
        masks_b = mask
        imgs1 = gt
        imgs2 = gt
        msrloss = 8 * self.l1((1-mask)*output, (1-mask)*gt) + 0.8*self.l1(mask*output, mask*gt)
                  #6 * self.l1((1-masks_b)*inter2, (1-masks_b)*imgs2)+1*self.l1(masks_b*inter2, masks_b*imgs2)+ \
                  #5 * self.l1((1-masks_a)*inter1, (1-masks_a)*imgs1)+0.8*self.l1(masks_a*inter1, masks_a*imgs1)

        ### Perceptual Loss ###
        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)

        prcLoss = 0.0
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
            prcLoss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])
        ### Style Loss ###
        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(gram_matrix(feat_output[i]),
                                       gram_matrix(feat_gt[i]))
            styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[i]),
                                       gram_matrix(feat_gt[i]))
        GLoss = msrloss + holeLoss + validAreaLoss + prcLoss + styleLoss + mask_loss
        return GLoss.sum()


def dice_loss(input, target):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)

    input = input
    target = target

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram
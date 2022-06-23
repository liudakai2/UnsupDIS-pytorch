
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2

from utils.general import img_torch2numpy


class ComputeAlignLoss:
    def __init__(self, model):
        super(ComputeAlignLoss, self).__init__()
        
        self.hyp = model.hyp  # hyperparameters
        # [16., 4., 1.] or [1., 4., 16.]
        self.scales = [self.hyp['loss_scale1'], self.hyp['loss_scale2'], self.hyp['loss_scale3']]
    
    def __call__(self, pred, images):  # for consistency
        eps = 0.01
        warped_imgs, warped_ones = pred[1:]
        target_image, target_mask = images[:, :3, ...], images[:, 3:4, ...]
        target_mask = (target_mask > eps).expand(-1, 3, -1, -1)
        bs, device = images.shape[0], images.device
        
        loss_per_level = [torch.zeros(1, device=device) for _ in range(3)]
        for i, warped_img in enumerate(warped_imgs):
            warped_mask = (warped_ones[i] > eps).expand(-1, 3, -1, -1)
            if warped_mask.sum() == 0:
                # return None, None
                continue
            
            overlap_mask = target_mask & warped_mask
            loss_per_level[i] += F.l1_loss(warped_img[overlap_mask], target_image[overlap_mask])
            # loss_per_level[i] += F.l1_loss(warped_img * overlap_mask, target_image * overlap_mask)
        
        loss = sum([scale * loss_per_level[i] for i, scale in enumerate(self.scales)])
        return loss * bs, torch.cat((*loss_per_level, loss)).detach()
    

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # vgg_features = torchvision.models.vgg16(pretrained=True).features.eval().to(device)
        vgg_features = torchvision.models.vgg19(pretrained=True).features.eval().to(device)
        # vgg16: 4, 9, 16, 23, 30; vgg19: 4, 9, 18, 27, 36
        # blocks = [vgg_features[:16], vgg_features[16:30]]
        blocks = [vgg_features[:18], vgg_features[18:36]]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        
        self.resize = torch.nn.functional.interpolate if resize else None
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])

    def forward(self, x, y, mask=None):
        if mask is None:
            mask = torch.ones_like(x[:, :1, :, :])
        x, y = x * mask, y * mask
        if self.resize:
            x = self.resize(x, mode='bilinear', size=(224, 224), align_corners=False)
            y = self.resize(y, mode='bilinear', size=(224, 224), align_corners=False)
        x = self.normalize(x).float()
        y = self.normalize(y).float()
        
        losses = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            losses.append(F.mse_loss(x, y))
            
            # msk = self.resize(mask, mode='nearest', size=x.shape[-2:]) > 0
            # ch = x.shape[1]
            # msk = msk.expand(-1, ch, -1, -1)
            # losses.append(F.mse_loss(x[msk], y[msk]))
            
        return losses


class SeamMaskExtractor(object):
    def __init__(self, device):
        sobel_x = np.array([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], dtype=np.float32)
        sobel_y = np.array([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]], dtype=np.float32)
        ones = np.ones((3, 3), dtype=np.float32)
        kernels = []
        for kernel in [sobel_x, sobel_y, ones]:
            kernel = np.reshape(kernel, (1, 1, 3, 3))
            kernels.append(torch.from_numpy(kernel).to(device))
        self.edge_kernel_x, self.edge_kernel_y, self.seam_kernel = kernels
    
    @torch.no_grad()
    def __call__(self, mask):
        # shape(b,1,h,w)
        assert isinstance(mask, torch.Tensor) and len(mask.shape) == 4 and mask.size(1) == 1
        if self.edge_kernel_x.dtype != mask.dtype:
            self.edge_kernel_x = self.edge_kernel_x.type_as(mask)
            self.edge_kernel_y = self.edge_kernel_y.type_as(mask)
            self.seam_kernel = self.seam_kernel.type_as(mask)
        
        mask_dx = F.conv2d(mask, self.edge_kernel_x, bias=None, stride=1, padding=1).abs()
        mask_dy = F.conv2d(mask, self.edge_kernel_y, bias=None, stride=1, padding=1).abs()
        edge = (mask_dx + mask_dy).clamp_(0, 1)
        for _ in range(3):  # dilate
            edge = F.conv2d(edge, self.seam_kernel, bias=None, stride=1, padding=1).clamp_(0, 1)
            
        return edge


class ComputeFuseLoss:
    def __init__(self, model):
        super(ComputeFuseLoss, self).__init__()
        
        device = next(model.parameters()).device  # get model device
        self.hyp = model.hyp  # hyperparameters
        
        self.ploss = VGGPerceptualLoss(device)
        self.seam_extractor = SeamMaskExtractor(device)

    def __call__(self, stitched, images):  # predictions, targets, model
        assert len(stitched) == 2
        eps = 0.01
        stitched_lr, stitched_hr = stitched
        stride = stitched_hr.shape[-1] // stitched_lr.shape[-1]
        device = images.device
        lcontent_lr, lseam_lr = torch.zeros(1, device=device), torch.zeros(1, device=device)
        lcontent_hr, lseam_hr = torch.zeros(1, device=device), torch.zeros(1, device=device)
        lconsistency = torch.zeros(1, device=device)

        image1, mask1, image2, mask2 = torch.split(images, [3, 1, 3, 1], dim=1)  # channel dimension
        mask1, mask2 = (mask1 > eps).int().type_as(images), (mask2 > eps).int().type_as(images)  # binarize
        seam1, seam2 = self.seam_extractor(mask1), self.seam_extractor(mask2)
        seam_mask1, seam_mask2 = (mask1 * seam2).expand(-1, 3, -1, -1), (mask2 * seam1).expand(-1, 3, -1, -1)
        image1_lr, image2_lr = self.downsample(image1, mode='bilinear', stride=stride), \
                               self.downsample(image2, mode='bilinear', stride=stride)
        mask1_lr, mask2_lr = self.downsample(mask1, mode='nearest', stride=stride), \
                             self.downsample(mask2, mode='nearest', stride=stride)  # TODO: nearest or bilinear?
        seam_mask1_lr, seam_mask2_lr = self.downsample(seam_mask1, mode='nearest', stride=stride), \
                                       self.downsample(seam_mask2, mode='nearest', stride=stride)

        lcontent_lr += (self.ploss(stitched_lr, image1_lr, mask1_lr)[1] +
                        self.ploss(stitched_lr, image2_lr, mask2_lr)[1])
        lseam_lr += (F.l1_loss(stitched_lr[seam_mask1_lr > eps], image1_lr[seam_mask1_lr > eps]) +
                     F.l1_loss(stitched_lr[seam_mask2_lr > eps], image2_lr[seam_mask2_lr > eps]))

        lcontent_hr += (self.ploss(stitched_hr, image1, mask1)[0] +
                        self.ploss(stitched_hr, image2, mask2)[0])
        lseam_hr += (F.l1_loss(stitched_hr[seam_mask1 > eps], image1[seam_mask1 > eps]) +
                     F.l1_loss(stitched_hr[seam_mask2 > eps], image2[seam_mask2 > eps]))

        lconsistency += F.l1_loss(self.downsample(stitched_hr, mode='bilinear', stride=stride), stitched_lr)

        bs = image1.shape[0]
        loss = self.hyp['loss_lr'] * (lcontent_lr * self.hyp['cont_lr'] + lseam_lr * self.hyp['seam_lr']) + \
               self.hyp['loss_hr'] * (lcontent_hr * self.hyp['cont_hr'] + lseam_hr * self.hyp['seam_hr']) + \
               self.hyp['consistency'] * lconsistency
        return loss * bs, torch.cat((lcontent_lr, lseam_lr, lcontent_hr, lseam_hr, lconsistency, loss)).detach()
    
    @staticmethod
    def downsample(x, mode='nearest', stride=4):
        return F.interpolate(x, mode=mode, size=tuple([y // stride for y in x.shape[2:]]),
                             align_corners=False if mode == "bilinear" else None)
    
    @staticmethod
    def plot_seam(image, seam, tail='left'):
        image, seam = image.clone(), seam.clone()
        seam[seam > 0] = 1.
        image[:, :1, :, :] += seam[:, :1, :, :]
        image = image.clamp(0, 1)
        for i, image in enumerate(img_torch2numpy(image)):
            cv2.imwrite('tmp/%02d_%s_seam.png' % (i, tail), image)

import torch
import cv2
from utils.torch_utils import check_anomaly


class DLTSolver(object):
    def __init__(self):
        self.M1 = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32).unsqueeze(0)
        self.M2 = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32).unsqueeze(0)
        self.M3 = torch.tensor([
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1]], dtype=torch.float32).unsqueeze(0)
        self.M4 = torch.tensor([
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32).unsqueeze(0)
        self.M5 = torch.tensor([
            [0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32).unsqueeze(0)
        self.M6 = torch.tensor([
            [-1],
            [0],
            [-1],
            [0],
            [-1],
            [0],
            [-1],
            [0]], dtype=torch.float32).unsqueeze(0)
        self.M71 = torch.tensor([
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32).unsqueeze(0)
        self.M72 = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, -1, 0]], dtype=torch.float32).unsqueeze(0)
        self.M8 = torch.tensor([
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, -1]], dtype=torch.float32).unsqueeze(0)
        self.Mb = torch.tensor([
            [0, -1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32).unsqueeze(0)
    
    def solve(self, pred_4pt_shift, patch_size=128.):
        bs, device, dtype = pred_4pt_shift.size(0), pred_4pt_shift.device, pred_4pt_shift.dtype
        if isinstance(patch_size, float):
            p_width, p_height =  patch_size, patch_size
        else:
            p_width, p_height = patch_size
        pts_1_tile = torch.tensor([0., 0., p_width, 0., 0., p_height, p_width, p_height], dtype=torch.float32)
        pred_pt4 = pts_1_tile.reshape((8, 1)).unsqueeze(0).to(device).expand(bs, -1, -1)
        orig_pt4 = pred_4pt_shift + pred_pt4
        
        self.check_mat_device(device)
        # bs is dynamic
        A1 = torch.bmm(self.M1.expand(bs, -1, -1), orig_pt4)  # Column 1: [bs,8,8] x [bs,8,1] = [bs,8,1]
        A2 = torch.bmm(self.M2.expand(bs, -1, -1), orig_pt4)  # Column 2
        A3 = self.M3.expand(bs, -1, -1)  # Column 3: [bs, 8, 1]
        A4 = torch.bmm(self.M4.expand(bs, -1, -1), orig_pt4)  # Column 4
        A5 = torch.bmm(self.M5.expand(bs, -1, -1), orig_pt4)  # Column 5
        A6 = self.M6.expand(bs, -1, -1)  # Column 6
        A7 = torch.bmm(self.M71.expand(bs, -1, -1), pred_pt4) * torch.bmm(self.M72.expand(bs, -1, -1), orig_pt4)  # Column 7
        A8 = torch.bmm(self.M71.expand(bs, -1, -1), pred_pt4) * torch.bmm(self.M8.expand(bs, -1, -1), orig_pt4)  # Column 8
        
        A_mat = torch.cat((A1, A2, A3, A4, A5, A6, A7, A8), dim=-1)  # [bs,8,8]
        b_mat = torch.bmm(self.Mb.expand(bs, -1, -1), pred_pt4)  # [bs,8,1]
        # Solve the Ax = b
        H_8el = torch.linalg.solve(A_mat.float(), b_mat.float()).type(dtype).squeeze(-1)  # [bs,8]
        
        h_ones = torch.ones((bs, 1)).to(device).type_as(H_8el)
        H_mat = torch.cat((H_8el, h_ones), dim=1).reshape(-1, 3, 3)  # [bs, 3, 3]
        
        return H_mat
    
    def check_mat_device(self, device):
        if self.M1.device != device:  # stupid
            self.M1 = self.M1.to(device)
            self.M2 = self.M2.to(device)
            self.M3 = self.M3.to(device)
            self.M4 = self.M4.to(device)
            self.M5 = self.M5.to(device)
            self.M6 = self.M6.to(device)
            self.M71 = self.M71.to(device)
            self.M72 = self.M72.to(device)
            self.M8 = self.M8.to(device)
            self.Mb = self.Mb.to(device)


def STN(image2_tensor, H_tf, offsets=()):
    """Spatial Transformer Layer"""

    def _repeat(x, n_repeats):
        rep = torch.ones(1, n_repeats, dtype=x.dtype)
        x = torch.mm(x.reshape(-1, 1), rep)
        return x.reshape(-1)

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch, channels, height, width = im.shape
        device = im.device
    
        x, y = x.float().to(device), y.float().to(device)
        height_f, width_f = torch.tensor(height).float(), torch.tensor(width).float()
        out_height, out_width = out_size
    
        # scale indices from [-1, 1] to [0, width/height]
        # effect values will exceed [-1, 1], so clamp is unnecessary or even incorrect
        x = (x + 1.0) * width_f / 2.0
        y = (y + 1.0) * height_f / 2.0
    
        # do sampling
        x0 = x.floor().int()
        x1 = x0 + 1
        y0 = y.floor().int()
        y1 = y0 + 1
    
        x0 = torch.clamp(x0, 0, width - 1)
        x1 = torch.clamp(x1, 0, width - 1)
        y0 = torch.clamp(y0, 0, height - 1)
        y1 = torch.clamp(y1, 0, height - 1)
    
        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(num_batch) * dim1, out_height * out_width).to(device)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
    
        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.permute(0, 2, 3, 1).reshape(-1, channels).float()
        Ia, Ib, Ic, Id = im_flat[idx_a], im_flat[idx_b], im_flat[idx_c], im_flat[idx_d]
    
        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
    
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
    
        return output  # .clamp(0., 1.) stupid

    def _meshgrid(height, width):
        x_t = torch.mm(torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).unsqueeze(0))
        y_t = torch.mm(torch.linspace(-1.0, 1.0, height).unsqueeze(1), torch.ones(1, width))
    
        x_t_flat = x_t.reshape(1, -1)
        y_t_flat = y_t.reshape(1, -1)
    
        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
        return grid

    bs, nc, height, width = image2_tensor.shape
    device = image2_tensor.device

    is_nan = torch.isnan(H_tf.view(bs, 9)).any(dim=1)
    assert is_nan.sum() == 0, f'{image2_tensor.shape} {len(offsets)}, {[off.view(-1, 8)[is_nan] for off in offsets]}'
    H_tf = H_tf.reshape(-1, 3, 3).float()
    # grid of (x_t, y_t, 1)
    grid = _meshgrid(height, width).unsqueeze(0).expand(bs, -1, -1).to(device)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = torch.bmm(H_tf, grid)  # [bs,3,3] x [bs,3,w*h] -> [bs,3,w*h]
    x_s, y_s, t_s = torch.chunk(T_g, 3, dim=1)
    # The problem may be here as a general homo does not preserve the parallelism
    # while an affine transformation preserves it.
    t_s_flat = t_s.reshape(-1)
    eps, maximal = 1e-2, 10.
    t_s_flat[t_s_flat.abs() < eps] = eps
    # 1.25000 / 1.38283e-05 = inf   in float16 (6.55e4)

    #  batchsize * width * height
    x_s_flat = x_s.reshape(-1) / t_s_flat
    y_s_flat = y_s.reshape(-1) / t_s_flat

    input_transformed = _interpolate(image2_tensor, x_s_flat, y_s_flat, (height, width))

    output = input_transformed.reshape(bs, height, width, nc).permute(0, 3, 1, 2)
    check_anomaly(output, prefix='transformed feature map', _exit=True)

    return output


def Stitching_Domain_STN(inputs, size_tensor, resized_shift):
    """Stitching Domain Spatial Transformer Layer"""
    
    def _repeat(x, n_repeats):
        rep = torch.ones(1, n_repeats, dtype=x.dtype)
        x = torch.mm(x.reshape(-1, 1), rep)
        return x.reshape(-1)
    
    def _interpolate(im, x, y, out_size, size_tensor):
        # constants
        num_batch, channels, height, width = im.shape
        device = im.device
        
        x, y = x.float().to(device), y.float().to(device)
        height_f, width_f = torch.tensor(height).float(), torch.tensor(width).float()
        out_height, out_width = out_size
        
        # scale indices from [-1, 1] to [0, width/height]
        # effect values will exceed [-1, 1], so clamp is unnecessary or even incorrect
        # x = (x + 1.0) * width_f / 2.0
        # y = (y + 1.0) * height_f / 2.0
        x = x / (size_tensor[0] - 1) * size_tensor[0]
        y = y / (size_tensor[1] - 1) * size_tensor[1]
        
        # do sampling
        x0 = x.floor().int()
        x1 = x0 + 1
        y0 = y.floor().int()
        y1 = y0 + 1
        
        x0 = torch.clamp(x0, 0, width - 1)
        x1 = torch.clamp(x1, 0, width - 1)
        y0 = torch.clamp(y0, 0, height - 1)
        y1 = torch.clamp(y1, 0, height - 1)
        
        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(num_batch) * dim1, out_height * out_width).to(device)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
        
        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.permute(0, 2, 3, 1).reshape(-1, channels).float()
        Ia, Ib, Ic, Id = im_flat[idx_a], im_flat[idx_b], im_flat[idx_c], im_flat[idx_d]
        
        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        
        return output  # .clamp(0., 1.) stupid
    
    def _meshgrid(width_max, width_min, height_max, height_min):
        width, height = (width_max - width_min + 1).long(), (height_max - height_min + 1).long()
        x_t = torch.mm(torch.ones(height, 1), torch.linspace(width_min, width_max, width).unsqueeze(0))
        y_t = torch.mm(torch.linspace(height_min, height_max, height).unsqueeze(1), torch.ones(1, width))
        
        x_t_flat = x_t.reshape(1, -1)
        y_t_flat = y_t.reshape(1, -1)
        
        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
        return grid
    
    def _transform(image_tf, H_tf, width_max, width_min, height_max, height_min, size_tensor):
        bs, nc, height, width = image_tf.shape
        device = image_tf.device
        
        H_tf = H_tf.reshape(-1, 3, 3).float()
        # grid of (x_t, y_t, 1)
        out_width = (width_max - width_min + 1).long()
        out_height = (height_max - height_min + 1).long()
        grid = _meshgrid(width_max, width_min, height_max, height_min).unsqueeze(0).expand(bs, -1, -1).to(device)
        
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = torch.bmm(H_tf, grid)  # [bs,3,3] x [bs,3,w*h] -> [bs,3,w*h]
        x_s, y_s, t_s = torch.chunk(T_g, 3, dim=1)
        # The problem may be here as a general homo does not preserve the parallelism
        # while an affine transformation preserves it.
        t_s_flat = t_s.reshape(-1)
        eps, maximal = 1e-2, 10.
        t_s_flat[t_s_flat.abs() < eps] = eps
        # 1.25000 / 1.38283e-05 = inf   in float16 (6.55e4)
        
        #  batchsize * width * height
        x_s_flat = x_s.reshape(-1) / t_s_flat
        y_s_flat = y_s.reshape(-1) / t_s_flat
        
        input_transformed = _interpolate(image_tf, x_s_flat, y_s_flat, (out_height, out_width), size_tensor)
        
        output = input_transformed.reshape(bs, out_height, out_width, nc).permute(0, 3, 1, 2)
        
        return output

    bs, device, dtype = resized_shift.size(0), resized_shift.device, resized_shift.dtype
    vertices = torch.tensor([0, 0, 1, 0, 0, 1, 1, 1], device=device, dtype=dtype)
    pts_1 = (vertices * (size_tensor - 1).repeat(4)).reshape(1, 8, 1)
    pts_2 = pts_1 + resized_shift
    pts = torch.cat((pts_1, pts_2), dim=0).reshape(8, 2)
    pts_x, pts_y = pts.T
    width_max, width_min, height_max, height_min = pts_x.max().ceil(), pts_x.min().floor(), pts_y.max().ceil(), pts_y.min().floor()
    out_width = (width_max - width_min + 1).long().item()
    out_height = (height_max - height_min + 1).long().item()
    
    org, dst = pts[:4].float().cpu().numpy(), pts[4:].cpu().numpy()
    # H_tf = cv2.getPerspectiveTransform(org, dst)
    H_tf = cv2.getPerspectiveTransform(dst, org)  # I don't understand
    H_tf = torch.from_numpy(H_tf).float().to(device)

    img1, img2 = inputs.chunk(2, dim=1)  # pure image
    img1, img2 = torch.cat((img1, torch.ones_like(img1[:, :1, :, :])), dim=1), \
                 torch.cat((img2, torch.ones_like(img2[:, :1, :, :])), dim=1)  # image with mask
    # H_one = torch.eye(3, dtype=dtype, device=device)
    # img1_tf = _transform(img1, H_one, width_max, width_min, height_max, height_min, size_tensor)
    pad = [(0 - width_min).long(), (width_max - size_tensor[0] + 1).long(), (0 - height_min).long(), (height_max - size_tensor[1] + 1).long()]
    img1_tf = torch.nn.functional.pad(img1, pad)
    img2_tf = _transform(img2, H_tf, width_max, width_min, height_max, height_min, size_tensor)
    
    output = torch.cat((img1_tf, img2_tf), dim=1)

    # Regularization. I think `crop` is more proper than `resize`.
    # resized_height = out_height - out_height%8
    # resized_width = out_width - out_width%8
    dx, dy = (out_width % 8) // 2, (out_height % 8) // 2
    output = output[..., dy:out_height-dy, dx:out_width-dx]
    
    return output

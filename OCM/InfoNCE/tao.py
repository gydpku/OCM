import math
import numbers
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

if torch.__version__ >= '1.4.0':
    kwargs = {'align_corners': False}
else:
    kwargs = {}


def rgb2hsv(rgb):
    """Convert a 4-d RGB tensor to the HSV counterpart.

    Here, we compute hue using atan2() based on the definition in [1],
    instead of using the common lookup table approach as in [2, 3].
    Those values agree when the angle is a multiple of 30°,
    otherwise they may differ at most ~1.2°.

    References
    [1] https://en.wikipedia.org/wiki/Hue
    [2] https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    [3] https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L212
    """

    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]
    delta = Cmax - Cmin

    hue = torch.atan2(math.sqrt(3) * (g - b), 2 * r - g - b)
    hue = (hue % (2 * math.pi)) / (2 * math.pi)
    saturate = delta / Cmax
    value = Cmax
    hsv = torch.stack([hue, saturate, value], dim=1)
    hsv[~torch.isfinite(hsv)] = 0.
    return hsv


def hsv2rgb(hsv):
    """Convert a 4-d HSV tensor to the RGB counterpart.

    >>> %timeit hsv2rgb(hsv)
    2.37 ms ± 13.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    >>> %timeit rgb2hsv_fast(rgb)
    298 µs ± 542 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> torch.allclose(hsv2rgb(hsv), hsv2rgb_fast(hsv), atol=1e-6)
    True

    References
    [1] https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB_alternative
    """
    h, s, v = hsv[:, [0]], hsv[:, [1]], hsv[:, [2]]
    c = v * s

    n = hsv.new_tensor([5, 3, 1]).view(3, 1, 1)
    k = (n + h * 6) % 6
    t = torch.min(k, 4 - k)
    t = torch.clamp(t, 0, 1)

    return v - c * t


class RandomResizedCropLayer(nn.Module):
    def __init__(self, size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        '''
            Inception Crop
            size (tuple): size of fowarding image (C, W, H)
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        '''
        super(RandomResizedCropLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.size = size
        self.register_buffer('_eye', _eye)
        self.scale = scale
        self.ratio = ratio

    def forward(self, inputs, whbias=None):
        _device = inputs.device
        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)

        if whbias is None:
            whbias = self._sample_latent(inputs)

        _theta[:, 0, 0] = whbias[:, 0]
        _theta[:, 1, 1] = whbias[:, 1]
        _theta[:, 0, 2] = whbias[:, 2]
        _theta[:, 1, 2] = whbias[:, 3]

        grid = F.affine_grid(_theta, inputs.size(), **kwargs).to(_device)
        output = F.grid_sample(inputs, grid, padding_mode='reflection', **kwargs)

        if self.size is not None:
            output = F.adaptive_avg_pool2d(output, self.size)

        return output#再次仿射取样，——theta考虑whbias

    def _clamp(self, whbias):

        w = whbias[:, 0]
        h = whbias[:, 1]
        w_bias = whbias[:, 2]
        h_bias = whbias[:, 3]

        # Clamp with scale
        w = torch.clamp(w, *self.scale)
        h = torch.clamp(h, *self.scale)

        # Clamp with ratio
        w = self.ratio[0] * h + torch.relu(w - self.ratio[0] * h)
        w = self.ratio[1] * h - torch.relu(self.ratio[1] * h - w)

        # Clamp with bias range: w_bias \in (w - 1, 1 - w), h_bias \in (h - 1, 1 - h)
        w_bias = w - 1 + torch.relu(w_bias - w + 1)
        w_bias = 1 - w - torch.relu(1 - w - w_bias)

        h_bias = h - 1 + torch.relu(h_bias - h + 1)
        h_bias = 1 - h - torch.relu(1 - h - h_bias)

        whbias = torch.stack([w, h, w_bias, h_bias], dim=0).t()

        return whbias

    def _sample_latent(self, inputs):

        _device = inputs.device
        N, _, width, height = inputs.shape

        # N * 10 trial
        area = width * height
        target_area = np.random.uniform(*self.scale, N * 10) * area
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = np.exp(np.random.uniform(*log_ratio, N * 10))

        # If doesn't satisfy ratio condition, then do central crop
        w = np.round(np.sqrt(target_area * aspect_ratio))
        h = np.round(np.sqrt(target_area / aspect_ratio))
        cond = (0 < w) * (w <= width) * (0 < h) * (h <= height)
        w = w[cond]
        h = h[cond]
        cond_len = w.shape[0]
        if cond_len >= N:
            w = w[:N]
            h = h[:N]
        else:
            w = np.concatenate([w, np.ones(N - cond_len) * width])
            h = np.concatenate([h, np.ones(N - cond_len) * height])

        w_bias = np.random.randint(w - width, width - w + 1) / width
        h_bias = np.random.randint(h - height, height - h + 1) / height
        w = w / width
        h = h / height

        whbias = np.column_stack([w, h, w_bias, h_bias])
        whbias = torch.tensor(whbias, device=_device)

        return whbias


class HorizontalFlipRandomCrop(nn.Module):
    def __init__(self, max_range):
        super(HorizontalFlipRandomCrop, self).__init__()
        self.max_range = max_range
        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

    def forward(self, input, sign=None, bias=None, rotation=None):
        _device = input.device
        N = input.size(0)
        _theta = self._eye.repeat(N, 1, 1)

        if sign is None:
            sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        if bias is None:
            bias = torch.empty((N, 2), device=_device).uniform_(-self.max_range, self.max_range)
        _theta[:, 0, 0] = sign
        _theta[:, :, 2] = bias

        if rotation is not None:
            _theta[:, 0:2, 0:2] = rotation

        grid = F.affine_grid(_theta, input.size(), **kwargs).to(_device)
        output = F.grid_sample(input, grid, padding_mode='reflection', **kwargs)

        return output

    def _sample_latent(self, N, device=None):
        sign = torch.bernoulli(torch.ones(N, device=device) * 0.5) * 2 - 1
        bias = torch.empty((N, 2), device=device).uniform_(-self.max_range, self.max_range)
        return sign, bias


class Rotation(nn.Module):
    def __init__(self, max_range = 4):
        super(Rotation, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):
        _device = input.device
        #print(self.prob)
        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)#随机四个里生成一个数

            output = torch.rot90(input, aug_index, (2, 3))#如果是aug》0，从y轴转向x轴，转90*aug，反之亦然。（2，3）是要转的维度

            _prob = input.new_full((input.size(0),), self.prob)#产生一个inputsize大小，值为0.5的tensor，不会加在a上，直接给prob
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)#按照prob中p用beinoulli生成0/1值，实际上是每个样本是否输出的mask
            output = _mask * input + (1-_mask) * output#这样做要么是原图像，要么旋转90*aug

        else:
            aug_index = aug_index % self.max_range
            output = torch.rot90(input, aug_index, (2, 3))#旋转角度不mask，原样返回

        return output


class CutPerm(nn.Module):
    def __init__(self, max_range = 4):
        super(CutPerm, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):
        _device = input.device

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)

            output = self._cutperm(input, aug_index)

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = self._cutperm(input, aug_index)

        return output

    def _cutperm(self, inputs, aug_index):

        _, _, H, W = inputs.size()
        h_mid = int(H / 2)
        w_mid = int(W / 2)

        jigsaw_h = aug_index // 2
        jigsaw_v = aug_index % 2

        if jigsaw_h == 1:
            inputs = torch.cat((inputs[:, :, h_mid:, :], inputs[:, :, 0:h_mid, :]), dim=2)
        if jigsaw_v == 1:
            inputs = torch.cat((inputs[:, :, :, w_mid:], inputs[:, :, :, 0:w_mid]), dim=3)

        return inputs


class HorizontalFlipLayer(nn.Module):
    def __init__(self):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(HorizontalFlipLayer, self).__init__()

        _eye = torch.eye(2, 3)#对角矩阵取前两行
        self.register_buffer('_eye', _eye)

    def forward(self, inputs):
        _device = inputs.device

        N = inputs.size(0)#batch——size
        _theta = self._eye.repeat(N, 1, 1)#重复N份，拼一起
        r_sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1#0.5概率生成mask
        _theta[:, 0, 0] = r_sign#把mask加入
        grid = F.affine_grid(_theta, inputs.size(), **kwargs).to(_device)
        inputs = F.grid_sample(inputs, grid, padding_mode='reflection', **kwargs)

        return inputs#做一系列仿射变换，得到图像


class RandomColorGrayLayer(nn.Module):
    def __init__(self, p):
        super(RandomColorGrayLayer, self).__init__()
        self.prob = p#0.2

        _weight = torch.tensor([[0.299, 0.587, 0.114]])
        self.register_buffer('_weight', _weight.view(1, 3, 1, 1))

    def forward(self, inputs, aug_index=None):

        if aug_index == 0:
            return inputs

        l = F.conv2d(inputs, self._weight)#卷积处理，只有一个轨道了
        gray = torch.cat([l, l, l], dim=1)#通道扩增3倍，得到原来的大小

        if aug_index is None:
            _prob = inputs.new_full((inputs.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)

            gray = inputs * (1 - _mask) + gray * _mask

        return gray


class ColorJitterLayer(nn.Module):
    def __init__(self, p, brightness, contrast, saturation, hue):
        super(ColorJitterLayer, self).__init__()
        self.prob = p#0.8
        self.brightness = self._check_input(brightness, 'brightness')#[0.6,1.4]
        self.contrast = self._check_input(contrast, 'contrast')#[0.6,1.4]
        self.saturation = self._check_input(saturation, 'saturation')#[0.6,1.4]
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)#hue 0.8,return[-0.1,0.1]

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]#hue[-0.1,0.1]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_contrast(self, x):
        if self.contrast:
            factor = x.new_empty(x.size(0), 1, 1, 1).uniform_(*self.contrast)#
            means = torch.mean(x, dim=[2, 3], keepdim=True)#【batch——size，3，1，1】
            x = (x - means) * factor + means#【32】【3】每个先减去对应means，再【32】乘以一个【0.6到1.4】中对应数，然后加（1-factor）*means 也是对应【32】加
        return torch.clamp(x, 0, 1)#维持在0，1中

    def adjust_hsv(self, x):
        f_h = x.new_zeros(x.size(0), 1, 1)
        f_s = x.new_ones(x.size(0), 1, 1)
        f_v = x.new_ones(x.size(0), 1, 1)#生成（batch_size,1,1）的0/1矩阵

        if self.hue:
            f_h.uniform_(*self.hue)#生成【batch_size,1,1】其中值在-0.1，0.1之间
        if self.saturation:
            f_s = f_s.uniform_(*self.saturation)#同事，值在0.6到1.4之间
        if self.brightness:
            f_v = f_v.uniform_(*self.brightness)

        return RandomHSVFunction.apply(x, f_h, f_s, f_v)#对每个通道做一些随机HSV变化

    def transform(self, inputs):
        # Shuffle transform
        if np.random.rand() > 0.5:
            transforms = [self.adjust_contrast, self.adjust_hsv]
        else:
            transforms = [self.adjust_hsv, self.adjust_contrast]

        for t in transforms:
            inputs = t(inputs)#对input随机套两个组合比较是必须的

        return inputs

    def forward(self, inputs):
        _prob = inputs.new_full((inputs.size(0),), self.prob)
        _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)#生成mask
        return inputs * (1 - _mask) + self.transform(inputs) * _mask


class RandomHSVFunction(Function):
    @staticmethod
    def forward(ctx, x, f_h, f_s, f_v):
        # ctx is a context object that can be used to stash information
        # for backward computation
        x = rgb2hsv(x)#从 hsv tensor 变 RGB tensor
        h = x[:, 0, :, :]#第一个通道【32，32，32】
        h += (f_h * 255. / 360.)#给每个在【32】中的值加f_h*255/360 对应的那个位置的值
        h = (h % 1)#求余数
        x[:, 0, :, :] = h#第一个通道这样，加法然后取余
        x[:, 1, :, :] = x[:, 1, :, :] * f_s#这里只是乘
        x[:, 2, :, :] = x[:, 2, :, :] * f_v
        x = torch.clamp(x, 0, 1)#裁剪，超过0，1范围的变0/1
        x = hsv2rgb(x)#返回
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
        return grad_input, None, None, None


class NormalizeLayer(nn.Module):
    """
    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """

    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, inputs):
        return (inputs - 0.5) / 0.5

import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.nn.functional import conv2d, pad as torch_pad
from typing import Any, List, Sequence, Optional
import numbers
import numpy as np
import torch
from PIL import Image
from typing import Tuple

class GaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.
    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.
        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.
        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.
        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return gaussian_blur(img, self.kernel_size, [sigma, sigma])

    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s

@torch.jit.unused
def _is_pil_image(img: Any) -> bool:
    return isinstance(img, Image.Image)
def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size
def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2
def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d

def _cast_squeeze_in(img: Tensor, req_dtype: torch.dtype) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype != req_dtype:
        need_cast = True
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype
def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        # it is better to round before cast
        img = torch.round(img).to(out_dtype)

    return img
def _get_gaussian_kernel2d(
        kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d
def _gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    """PRIVATE METHOD. Performs Gaussian blurring on the img by given kernel.
    .. warning::
        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.
    Args:
        img (Tensor): Image to be blurred
        kernel_size (sequence of int or int): Kernel size of the Gaussian kernel ``(kx, ky)``.
        sigma (sequence of float or float, optional): Standard deviation of the Gaussian kernel ``(sx, sy)``.
    Returns:
        Tensor: An image that is blurred using gaussian kernel of given parameters
    """
    if not (isinstance(img, torch.Tensor) or _is_tensor_a_torch_image(img)):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, kernel.dtype)

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img

def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None) -> Tensor:
    """Performs Gaussian blurring on the img by given kernel.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    Args:
        img (PIL Image or Tensor): Image to be blurred
        kernel_size (sequence of ints or int): Gaussian kernel size. Can be a sequence of integers
            like ``(kx, ky)`` or a single integer for square kernels.
            In torchscript mode kernel_size as single int is not supported, use a tuple or
            list of length 1: ``[ksize, ]``.
        sigma (sequence of floats or float, optional): Gaussian kernel standard deviation. Can be a
            sequence of floats like ``(sigma_x, sigma_y)`` or a single float to define the
            same sigma in both X/Y directions. If None, then it is computed using
            ``kernel_size`` as ``sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8``.
            Default, None. In torchscript mode sigma as single float is
            not supported, use a tuple or list of length 1: ``[sigma, ]``.
    Returns:
        PIL Image or Tensor: Gaussian Blurred version of the image.
    """
    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError('kernel_size should be int or a sequence of integers. Got {}'.format(type(kernel_size)))
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError('If kernel_size is a sequence its length should be 2. Got {}'.format(len(kernel_size)))
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError('kernel_size should have odd and positive integers. Got {}'.format(kernel_size))

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError('sigma should be either float or sequence of floats. Got {}'.format(type(sigma)))
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError('If sigma is a sequence, its length should be 2. Got {}'.format(len(sigma)))
    for s in sigma:
        if s <= 0.:
            raise ValueError('sigma should have positive values. Got {}'.format(sigma))

    t_img = img
    if not isinstance(img, torch.Tensor):
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image or Tensor. Got {}'.format(type(img)))

        t_img = to_tensor(img)

    output = _gaussian_blur(t_img, kernel_size, sigma)

    if not isinstance(img, torch.Tensor):
        output = to_pil_image(output)
    return output

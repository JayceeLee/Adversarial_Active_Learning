import numpy as np
import torch
import random
import cv2
from easydict import EasyDict as EDict

def random_contrast_brightness(img, max_rand_contrast, max_rand_brightness):
    # img_float = img.astype(float)
    contrast = random.random() * (1 + max_rand_contrast - (1 - max_rand_contrast)) + 1 - max_rand_contrast
    brightness = random.random() * (max_rand_brightness - (-max_rand_brightness)) + (-max_rand_brightness)
    out_img = np.clip(contrast * img + brightness, 0, 255)
    return out_img


def random_gaussian_smoothing(in_img, config):
    Ks = random.random() * (config.trans_photo.gauss_smooth_k_max - config.trans_photo.gauss_smooth_k_min) + config.trans_photo.gauss_smooth_k_min
    Ks = int(Ks)
    if Ks % 2 == 0:
        Ks += 1
    # Opencv uses this formula to compute sigma when only the kernel size is provided
    # Here a fixed size kernel is used instead of the randomly generated kernel size
    # sigma = 0.3 * ((Ks - 1) * 0.5 - 1) + 0.8
    blur = cv2.GaussianBlur(in_img, (Ks, Ks), 0, 0)
    return blur


def random_bilateral_smoothing(in_img):
    # See opencv documentation for the for parameter settings
    mind = 5
    maxd = 9
    d = int(random.random() * (maxd - mind) + mind)
    minSig = 40
    maxSig = 75
    sig = int(random.random() * (maxSig - minSig) + minSig)
    blur = cv2.bilateralFilter(in_img, d, sig, sig)
    return blur


def gaussian_noise(image, mean, sigma):
    image += np.random.normal(mean, sigma, image.shape)


def random_gaussian_noise(image, mean, sigma):
    sigma_ = random.uniform(sigma / 2.0, sigma)
    image += np.random.normal(mean, sigma_, image.shape)


def salt_pepper_noise(image, s_vs_p, ratio):
    img_shape = image.shape
    num_salt = int(np.prod(img_shape) * s_vs_p * ratio)
    num_pepper = int(np.prod(img_shape) * (1.0 - s_vs_p) * ratio)
    noise_indices = np.random.randint(0, np.prod(img_shape), num_salt + num_pepper)
    image = image.flatten()
    image[noise_indices[:num_salt]] = 1.0
    image[noise_indices[num_salt:]] = 0
    return image.reshape(img_shape)


def poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    return np.random.poisson(image * vals) / float(vals)


def speckle_noise(image, mean, sigma):
    image += image * np.random.normal(mean, sigma, image.shape)


def log_transform(img, max_=2.0, min_=1.0):
    lut = np.arange(256).astype(float)/255.0
    c = np.random.random_sample() * (max_ - min_) + min_
    lut = c*np.log(1+lut)*255
    cv2.normalize(lut, lut, 0, 255, cv2.NORM_MINMAX)
    lut = lut.astype('uint8')
    log_img = cv2.LUT(img, lut)
    return log_img


def contrast_stretch(img,max_=2.0, min_=1.0):
    lut = np.arange(256).astype(float)/255.0;
    E = np.random.random_sample()*(max_-min_)+min_
    lut = 255./(1 + np.power(.5/(lut + 1e-5),E))
    cv2.normalize(lut, lut, 0, 255, cv2.NORM_MINMAX)
    lut = lut.astype('uint8')
    cs_img = cv2.LUT(img, lut)
    return cs_img


def gamma_corr(img, max_=2.0, min_=0.9):
    lut = np.arange(256).astype(float) / 255.0
    gamma = np.random.random_sample()*(max_-min_)+min_
    lut = np.power(lut, gamma)*255
    cv2.normalize(lut, lut, 0, 255, cv2.NORM_MINMAX)
    lut = lut.astype('uint8')
    aug_img = cv2.LUT(img, lut)
    return aug_img


def aug_rescale(aug_patch, config):
    # pdb.set_trace()
    input_shape = aug_patch.shape[:2][::-1] #(WIDTH, HEIGHT)
    tmp = cv2.resize(aug_patch, (0, 0), fx=config.trans_photo.rescale_down, fy=config.trans_photo.rescale_down,
                     interpolation=cv2.INTER_AREA)
    toss = random.random()
    if toss < config.trans_photo.rescale_down_linear_upsample_prob:
        tmp = cv2.resize(tmp, tuple(input_shape), interpolation=cv2.INTER_LINEAR)
    else:
        tmp = cv2.resize(tmp, tuple(input_shape), interpolation=cv2.INTER_NEAREST)
    return tmp


def decrease_contrast(img, gauss_ksize):
    img_blur = cv2.GaussianBlur(img, (gauss_ksize, gauss_ksize), 0, 0)
    img = (img.astype(np.float32) / (img_blur.astype(np.float32) + 1.0))
    img = (img / img.max())*255.0
    return img

def get_edge_strength(img):
    dx = abs(cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=7))
    dy = abs(cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=7))
    # dx = cv2.convertScaleAbs(dx)
    # dy = cv2.convertScaleAbs(dy)
    temp = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
    temp = temp / temp.max()
    # plt.imshow(temp, cmap='gray')
    # plt.show()
    return temp * 255.0


def aug_photometric_eye_all(aug_patch, config):
    # aug_patch: uint8
    if 'smoothing_aug' in config.trans_photo and config.trans_photo.smoothing_aug:
        if random.random() < config.trans_photo.rescale_down_prob:
            aug_patch = aug_rescale(aug_patch, config) #unit8
        else:
            aug_patch = random_gaussian_smoothing(aug_patch, config) #uint8

    if 'noise_aug' in config.trans_photo and config.trans_photo.noise_aug:
        aug_patch = aug_patch.astype(np.float32) / 255.0 #float [0, 1]
        noise_toss = random.random()
        if noise_toss < config.trans_photo.poisson_noise_prob:
            aug_patch = poisson_noise(aug_patch) #float
        else:
            speckle_noise(aug_patch, config.trans_photo.speckle_noise_mean,
                          config.trans_photo.speckle_noise_sigma)

        random_gaussian_noise(aug_patch, config.trans_photo.gaussian_noise_mean,
                              config.trans_photo.gaussian_noise_sigma)
        min_val = aug_patch.min()
        aug_patch = (((aug_patch - min_val) / (aug_patch.max() - min_val)) * 255) #float [0, 255]

    aug_patch = aug_patch.astype(np.float32) #Float [0,255]
    aug_patch = random_contrast_brightness(aug_patch, config.trans_photo.max_rand_contrast,
                                           config.trans_photo.max_rand_brightness) #float [0, 255]

    if 'graylevel_aug' in config.trans_photo and config.trans_photo.graylevel_aug:
        aug_patch = aug_patch.astype(np.uint8)
        if random.random() < config.trans_photo.log_transform_prob:
            aug_patch = log_transform(aug_patch) #float [0, 255]
        elif random.random() < config.trans_photo.log_transform_prob + config.trans_photo.gamma_corr_prob:
            aug_patch = gamma_corr(aug_patch) #float [0, 255]
        else:
            aug_patch = contrast_stretch(aug_patch) # float[0, 255]
        aug_patch = aug_patch.astype(np.float32) / 255.0 #float[0, 1]
        min_val = aug_patch.min()
        aug_patch = (((aug_patch - min_val) / (aug_patch.max() - min_val)) * 255) #float #255
    if 'dec_contrast_aug' in config.trans_photo and config.trans_photo.dec_contrast_aug:
        aug_patch = decrease_contrast(aug_patch, config.trans_photo.dec_contrast_ksize) #Float [0,255]
    if config.edge_strength:
        aug_patch = aug_patch.astype(np.uint8)
        aug_patch = get_edge_strength(aug_patch)
    return aug_patch.astype(np.uint8)


class PhotometricTransform(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, image):
        '''
        :sample: A dictionary with UINT8 image
        :output: UINT8 photometricaly augmented image
        '''
        # image = sample['image']
        aug_image = aug_photometric_eye_all(image, self.config)
        return aug_image


#SAMPLE CONFIGURATION FOR PHOTOMETRIC AUGMENTATION
photometric_transform_config = EDict({
    'trans_photo': {
        'smoothing_aug': True,
        'rescale_down_prob': 0.2,
        'rescale_down': 0.5,
        'rescale_down_linear_upsample_prob': 1.0,
        'gauss_smooth_k_min': 2,
        'gauss_smooth_k_max': 5,
        'noise_aug': False,
        'poisson_noise_prob': 0.4,
        'speckle_noise_mean': 0,
        'speckle_noise_sigma': 0.025,
        'gaussian_noise_mean': 0,
        'gaussian_noise_sigma': 0.025,
        'graylevel_aug': True,
        'max_rand_contrast': 0.2,
        'max_rand_brightness': 15,
        'log_transform_prob': 0.2,
        'gamma_corr_prob': 0.2,
        'dec_contrast_aug': False,
        'dec_contrast_ksize': 5 #Increase it to odd numbers to reduce the effect of texture removal from the images
    },
    'edge_strength': False #Keep it false always
})

def __main__():
    #Creating a transformation class for pytorch transform class
    photo_transformer = PhotometricTransform(photometric_transform_config)

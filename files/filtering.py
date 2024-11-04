'''Filter'''
import numpy as np

def get_center(kernel):
    '''Function that gets the center of the kernel'''
    if kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1:
        return [kernel.shape[0]//2, kernel.shape[1]//2]
    return [0, 0]


def get_padding(center, kernel):
    '''Gets number of zeros to be padded in each direction'''
    if center == [0,0]:
        return ((1,1),(1,1))
    return (center[0], kernel.shape[0]-1-center[0]), (center[1], kernel.shape[1]-1-center[1])


def convolve(array, kernel):
    '''Applies convolution to a array'''
    center = get_center(kernel)
    padding = get_padding(center, kernel)
    pad_image = np.pad(array, padding)
    sha = (array.shape[0], array.shape[1], kernel.shape[0], kernel.shape[1])
    stride = np.lib.stride_tricks.as_strided(pad_image, shape=sha, strides=pad_image.strides*2)
    return np.sum(stride * kernel, axis=(2, 3))


def apply_filter(image: np.array, kernel: np.array) -> np.array:
    '''Applies filter on a image'''
    assert image.ndim in [2, 3]
    assert kernel.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]
    if len(image.shape) == 3:
        red = image[:, :, 0]
        blue = image[:, :, 1]
        green = image[:, :, 2]

        red = convolve(red, kernel)
        blue = convolve(blue, kernel)
        green = convolve(green, kernel)

        result = np.dstack((red, blue, green))
    else:
        result = convolve(image, kernel)

    result = np.clip(result, 0, 255)
    result = result.astype(np.uint8)
    return result

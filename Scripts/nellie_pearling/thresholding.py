import cupy as cp
import numpy as np
import scipy.ndimage as ndi


def triangle_threshold(matrix, nbins=256):
    # gpu version of skimage.filters.threshold_triangle
    hist, bin_edges = cp.histogram(matrix.reshape(-1), bins=nbins, range=(matrix.min(), matrix.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    hist = hist / cp.sum(hist)

    arg_peak_height = cp.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = cp.flatnonzero(hist)[[0, -1]]

    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = hist[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1
    del(arg_high_level)

    width = arg_peak_height - arg_low_level
    x1 = cp.arange(width)
    y1 = hist[x1 + arg_low_level]

    norm = cp.sqrt(peak_height**2 + width**2)
    peak_height = peak_height / norm
    width = width / norm

    length = peak_height * x1 - width * y1
    arg_level = cp.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return bin_centers[arg_level]


def otsu_threshold(matrix, nbins=256):
    # gpu version of skimage.filters.threshold_otsu
    counts, bin_edges = cp.histogram(matrix.reshape(-1), bins=nbins, range=(matrix.min(), matrix.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    counts = counts / cp.sum(counts)

    weight1 = cp.cumsum(counts)
    weight2 = cp.cumsum(counts[::-1])[::-1]
    mean1 = cp.cumsum(counts * bin_centers) / weight1
    mean2 = (cp.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = cp.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold


def remove_small_objects(mask, size_thresh):
    labels, _ = ndi.label(mask)
    areas = np.bincount(labels.ravel())
    mask_sizes = areas > size_thresh

    # only keep labels where mask sizes is true
    mask_sizes[0] = 0
    labels_cleaned, num_labels = ndi.label(mask_sizes[labels])
    return labels_cleaned, num_labels


def minotri_threshold(matrix):
    triangle_thresh = triangle_threshold(matrix)
    otsu_thresh = otsu_threshold(matrix)
    return min(triangle_thresh, otsu_thresh)

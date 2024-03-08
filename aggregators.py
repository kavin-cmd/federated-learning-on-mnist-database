from functools import partial

import numpy as np
import wquantiles as w

def mean(points, weights):
    return np.average(points, axis=0, weights=weights)#.astype(points.dtype)


def coordinatewise(fn, points, weights):
    points = np.asarray(points)
    if points.ndim == 1:
        return fn(points, weights)
    shape = points.shape
    res = np.empty_like(points, shape=shape[1:])
    for index in np.ndindex(*shape[1:]):
        coordinates = points[(..., *index)]
        res[index] = fn(coordinates, weights)
    return res


def quantile(points, weights, quantile):
    if weights is None:
        return np.quantile(points, quantile, axis=0).astype(np.float32)
    return coordinatewise(partial(w.quantile_1D, quantile=quantile), points, weights)


def median(points, weights):
    return quantile(points, weights, 0.5)
    # return np.median(points, axis=0) if weights is None \
    #     else np.apply_along_axis(weightedstats.numpy_weighted_median, 0,
    #                              points,
    #                              weights)


def trimmed_mean_1d(vector, weights, beta):
    if weights is None:
        lower_bound, upper_bound = np.quantile(vector, (beta, 1 - beta)).astype(np.float32)
        trimmed = [v for v in vector if lower_bound < v < upper_bound]
        if trimmed:
            return mean(trimmed, None)
        else:
            return (lower_bound + upper_bound) / 2
    else:
        lower_bound, upper_bound = w.quantile_1D(vector, weights, beta), w.quantile_1D(vector, weights, 1 - beta)

        trimmed = [(v, w) for v, w in zip(vector, weights) if lower_bound < v < upper_bound]
        if trimmed:
            trimmed_vector, trimmed_weights = zip(*trimmed)

            return mean(trimmed_vector, trimmed_weights)
        else:
            return (lower_bound + upper_bound) / 2


def trimmed_mean(points, weights, beta):
    return coordinatewise(partial(trimmed_mean_1d, beta=beta), points, weights)

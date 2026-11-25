"""
Temporally consistent multi-date classification (AI change detection).

Conventional post-classification comparison classifies each date on its own
and subtracts the maps. Every per-date error then shows up as "change",
which in our Guwahati test inflated change to 27% of the area.

GeoChange AI couples the two dates instead:
  1. Spectral change intensity: robust z-scored difference of all bands and
     indices between dates (median/MAD per scene, so leftover haze or
     illumination differences cancel out), smoothed 3x3.
  2. Automatic threshold: Otsu's method on the intensity histogram - no
     manual tuning per scene or per district.
  3. Pixels below the threshold are spectrally stable: they get ONE class for
     both dates, the argmax of the summed AI probabilities of both dates.
     Pixels above it keep their per-date class.

Result: a label can only change where the ground actually changed spectrally,
and stable land gets the more reliable two-date decision.
"""

import numpy as np
from scipy import ndimage as ndi

from modules.classification import majority_filter, predict_proba_stack
from config import LULC_CLASSES

N_SPECTRAL = 10  # bands + indices (texture excluded)


def otsu_threshold(values, bins=256):
    x = np.clip(values, 0, np.percentile(values, 99.5))
    hist, edges = np.histogram(x, bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * centers)
    total, mtotal = w0[-1], m0[-1]
    w1 = total - w0
    between = (mtotal * w0 / total - m0) ** 2 / (w0 * w1 + 1e-9)
    return float(centers[np.argmax(between)])


def spectral_change(stack_a, stack_b):
    d = stack_b[:N_SPECTRAL] - stack_a[:N_SPECTRAL]
    valid = np.isfinite(d).all(axis=0)
    z = np.zeros(d.shape[1:], dtype="float32")
    for band in d:
        v = band[valid]
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
        z += np.nan_to_num(((band - med) / mad) ** 2)
    mag = ndi.uniform_filter(np.sqrt(z / len(d)), 3)
    mag[~valid] = np.nan
    return mag, otsu_threshold(mag[valid])


def classify_multitemporal(model, stacks):
    """stacks: list of feature stacks [first, ..., last] on one grid.

    Returns (classes list, confidences list, change magnitude first->last, threshold).
    """
    shape = stacks[0].shape[1:]
    probas, valids = zip(*[predict_proba_stack(model, s) for s in stacks])
    mag, thr = spectral_change(stacks[0], stacks[-1])
    stable = (np.nan_to_num(mag, nan=np.inf) <= thr).ravel()

    joint = sum(probas) / len(probas)
    classes, confs = [], []
    for p, v in zip(probas, valids):
        q = np.where(stable[:, None], joint, p)
        cls = np.zeros(q.shape[0], dtype="uint8")
        cls[v] = model.classes_[q[v].argmax(axis=1)]
        conf = np.zeros(q.shape[0], dtype="uint8")
        conf[v] = np.round(q[v].max(axis=1) * 100).astype("uint8")
        classes.append(majority_filter(cls.reshape(shape), v.reshape(shape), len(LULC_CLASSES)))
        confs.append(conf.reshape(shape))
    return classes, confs, mag, thr

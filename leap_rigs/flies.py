"""Fly-specific functionality, such as feature extraction from pose."""

import numpy as np
from typing import Tuple
import attr


# Standard 13 node skeleton:
FLY_NODES = [
    "head",
    "thorax",
    "abdomen",
    "wingL",
    "wingR",
    "forelegL4",
    "forelegR4",
    "midlegL4",
    "midlegR4",
    "hindlegL4",
    "hindlegR4",
    "eyeL",
    "eyeR",
]
FLY_EDGES = [
    (1, 0),
    (0, 11),
    (0, 12),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 9),
    (1, 10),
]
FLY_HEAD_IND = FLY_NODES.index("head")
FLY_THORAX_IND = FLY_NODES.index("thorax")
FLY_ABDOMEN_IND = FLY_NODES.index("abdomen")

# Standard colors
COL_F = (0.7561707035755478, 0.21038062283737025, 0.22352941176470587)
COL_M = (0.1843137254901961, 0.47266435986159167, 0.7116493656286044)

# Calibration
PX_PER_MM = 30.3


def signed_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Find the signed angle between two 2D vectors a and b.

    Args:
        a: Array of shape (n, 2).
        b: Array of shape (n, 2).

    Returns:
        The signed angles in degrees in vector of shape (n, 2).

        This angle is positive if a is rotated clockwise to align to b and negative if
        this rotation is counter-clockwise.
    """
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    theta = np.arccos(np.around(np.sum(a * b, axis=1), decimals=4))
    cross = np.cross(a, b, axis=1)
    sign = np.zeros(cross.shape)
    sign[cross >= 0] = -1
    sign[cross < 0] = 1
    return np.rad2deg(theta) * sign


def hasnan(x) -> bool:
    """Return True if x contains any NaNs."""
    return np.isnan(x).any()


@attr.s(auto_attribs=True, slots=True)
class Features:
    """Features derived from poses.

    Attributes:
        dist: Distance between the thorax of the male and female in mm.
        min_dist: Distance between the head of the male and the abdomen of the female in
            mm.
        ang_f_rel_m: Angular location of the female thorax relative to the male. If this
            is negative, the female is to the LEFT of the male.
        ang_m_rel_f: Angular location of the male thorax relative to the female. If this
            is negative, the male is to the LEFT of the female.
    """

    dist: float = np.nan
    min_dist: float = np.nan
    ang_f_rel_m: float = np.nan
    ang_m_rel_f: float = np.nan


def compute_features(pose_f: np.ndarray, pose_m: np.ndarray) -> Features:
    """Compute relative features between the flies.

    Args:
        pose_f: Numpy array with female pose of shape (13, 2)
        pose_m: Numpy array with male pose of shape (13, 2)

    Returns:
        The relative features as a Features object.
    """
    ctr_f, head_f = pose_f[FLY_THORAX_IND], pose_f[FLY_HEAD_IND]
    ctr_m, head_m = pose_m[FLY_THORAX_IND], pose_m[FLY_HEAD_IND]
    abd_f = pose_f[FLY_ABDOMEN_IND]

    feats = Features()

    # Thorax-thorax distance
    if not (hasnan(ctr_f) or hasnan(ctr_m)):
        feats.dist = np.linalg.norm(ctr_f - ctr_m) / PX_PER_MM

    # Male head to female abdomen distance
    if not (hasnan(head_m) or hasnan(abd_f)):
        feats.min_dist = np.linalg.norm(head_m - abd_f) / PX_PER_MM

    # Angular location of the female relative to the male
    if not (hasnan(ctr_f) or hasnan(ctr_m) or hasnan(head_m)):
        feats.ang_f_rel_m = signed_angle(
            np.expand_dims(ctr_f - ctr_m, axis=0),
            np.expand_dims(head_m - ctr_m, axis=0),
        ).squeeze()

    # Angular location of the male relative to the female
    if not (hasnan(ctr_m) or hasnan(ctr_f) or hasnan(head_f)):
        feats.ang_m_rel_f = signed_angle(
            np.expand_dims(ctr_m - ctr_f, axis=0),
            np.expand_dims(head_f - ctr_f, axis=0),
        ).squeeze()

    return feats


class PoseBuffer:
    def __init__(self):
        self.last_pose_m = None
        self.last_pose_f = None
        self.pose_m = None
        self.pose_f = None

    def update(self, pred):
        self.pose_f, self.pose_m = pred["instance_peaks"][0]
        if self.last_pose_f is None or (~np.isnan(self.pose_f)).any():
            self.last_pose_f = self.pose_f
        if self.last_pose_m is None or (~np.isnan(self.pose_m)).any():
            self.last_pose_m = self.pose_m

    def compute_features(self):
        return compute_features(self.last_pose_f, self.last_pose_m)

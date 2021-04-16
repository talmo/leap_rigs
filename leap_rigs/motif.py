"""Motif setup and control via the API."""

import os
import json
import yaml
import threading
import motifapi
from motifapi import MotifApi, MotifError
import numpy as np
from time import perf_counter
from typing import Tuple, Dict


def get_motif_remote(api_key=None, ip=None) -> MotifApi:
    """Return a MotifApi object for communicating with the Motif server."""
    if (ip is not None) and (api_key is not None):
        return MotifApi(None, None)

    with open("C:\\ProgramData\\Motif\\recnode.yml", "rt") as f:
        conf = yaml.load(f)
        try:
            _key = conf["Common"]["APIKey"]
        except:
            _key = None

        try:
            _ip = conf["Common"]["NetworkIP"]
        except:
            _ip = "127.0.0.1"

        return MotifApi(
            host=ip if ip is not None else _ip,
            api_key=api_key if api_key is not None else _key,
        )


def get_experiment_metadata():
    """Read experiment metadata from the env variable MOTIF_METADATA_JSON_PATH."""
    try:
        mdf = os.environ["MOTIF_METADATA_JSON_PATH"]
        with open(mdf, "r") as f:
            return json.load(f)
    except KeyError:
        # not defined
        pass
    except Exception as exc:
        print("Error parsing metadata file", exc)

    return {}


class StreamPoller(threading.Thread):
    """Threaded poller for the latest camera image from a Motif stream.

    Attributes:
        api: The MotifApi object for communicating with Motif. This can be acquired by
            using get_motif_remote().
        camera_sn: Serial number for the camera to be polled.
    """

    daemon = True

    def __init__(self, api: MotifApi, camera_sn: str):
        super().__init__()

        print("Initializing image polling thread...")
        self.api = api
        self.camera_sn = camera_sn
        self._img = None
        self._md = None
        self._lock = threading.Lock()
        self._stream = api.get_stream(camera_sn, stream_type=MotifApi.STREAM_TYPE_IMAGE)
        assert self._stream is not None
        print(f"Got stream for image polling (camera: {camera_sn}).")

    def run(self):
        while True:
            I, md = self._stream.get_next_image(copy=False)  # warning: blocks!
            with self._lock:
                self._img = I.copy()
                self._md = md.copy()
                self._md["timestamp"] = perf_counter()

    @property
    def latest_image(self) -> Tuple[np.ndarray, Dict]:
        """Return latest image and metadata."""
        with self._lock:
            return self._img, self._md

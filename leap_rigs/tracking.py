"""Realtime SLEAP tracking for closed-loop experiments."""

from time import perf_counter, sleep
import cv2
import numpy as np
import threading
from typing import Tuple, Optional, Union, List, Dict, Callable, Any


class LivePredictor(threading.Thread):
    """Threaded SLEAP inference for asynchronous prediction.

    Attributes:
        predictor: A sleap.Predictor object with a loaded inference model.
        get_image_fn: A function that can be called to retrieve new data for inference.
            See the notes below for the expected return signature.

    Notes:
        The get_image_fn function will be called when this thread is running in order to
        get new images to use for inference.

        The function will be called with no inputs and should return a tuple of
        (img, metadata), where:

        img: A numpy array of dtype uint8 and shape (height, width, channels) or
        (1, height, width, channels).

        metadata: This can be anything and will be stored together with the inference
        results. This can be None if not needed.

        Example:
            def dummy_img_fn():
                return np.zeros((128, 128, 1), dtype="uint8"), None

            lp = LivePredictor.load_model("my_model", get_image_fn=dummy_img_fn)
    """

    daemon = True

    def __init__(
        self, predictor: "sleap.Predictor", get_image_fn: Optional[Callable] = None
    ):
        super().__init__()
        self.predictor = predictor
        self._lock = threading.Lock()
        self._last_prediction = None
        self._last_image = None
        self._last_meta = None
        self._last_timestamp = None
        self._last_latency = None
        self.get_image_fn = get_image_fn

    @classmethod
    def load_model(
        cls, model_path: Union[List[str], str], get_image_fn: Optional[Callable] = None
    ) -> "LivePredictor":
        """Load SLEAP model(s) and return a live predictor.

        Args:
            model_path: Path or paths to trained SLEAP models. See sleap.load_model()
                for more information on accepted inputs.
            get_image_fn: Function to use to fetch new images when running as a thread.
        """
        import sleap

        predictor = sleap.load_model(model_path, batch_size=1)
        return cls(predictor=predictor, get_image_fn=get_image_fn)

    @property
    def last_prediction(self) -> Dict[str, np.ndarray]:
        """Return the last prediction from the model as a dictionary of numpy arrays."""
        with self._lock:
            return self._last_prediction

    @property
    def last_data_and_prediction(
        self,
    ) -> Tuple[Tuple[np.ndarray, Optional[Any]], Tuple[Dict[str, np.ndarray], float]]:
        """Return the last image, metadata and prediction from the model.

        Returns:
            A tuple of (image, metadata), (prediction, timestamp).

            image: Copy of the image used as input.
            metadata: Any metadata that was passed in with the image.
            prediction: The output of the model as a dictionary of arrays.
            timestamp: Time that the prediction was generated.
        """
        with self._lock:
            return (self._last_image, self._last_meta), (
                self._last_prediction,
                self._last_timestamp,
            )

    def predict(
        self, image: np.ndarray, meta: Optional[Any] = None
    ) -> Dict[str, np.ndarray]:
        """Run inference and store results.

        Args:
            image: A rank-3 or rank-4 image to provide as input to the model.
            meta: Any metadata associated with this image.

        Returns:
            The model prediction as a dictionary of arrays.

        Notes:
            This saves both the inputs and the resulting outputs to thread-safe cached
            attributes which can be queried asynchronously.

            See last_prediction and last_data_and_prediction.
        """
        t0 = perf_counter()
        image = np.expand_dims(image, axis=-1) if image.ndim == 2 else image
        image = np.expand_dims(image, axis=0) if image.ndim == 3 else image
        pred = self.predictor.inference_model.predict_on_batch(image)
        latency = perf_counter() - t0
        with self._lock:
            self._last_image = image.copy()
            self._last_prediction = pred
            self._last_meta = meta
            self._last_timestamp = perf_counter()
            self._last_latency = latency
        return pred

    def run(self):
        """Run the realtime inference thread."""
        if self.get_image_fn is None:
            raise ValueError("Cannot start live prediction if get_image_fn is not set.")

        last_img = None
        while True:
            # Get a new image.
            img, meta = self.get_image_fn()

            if img is None:
                continue
            #     # Received poison pill, so we're finished.
            #     break

            # Check if this is a new image.
            # TODO: Compare by metadata could be faster?
            is_new_img = last_img is None or ~np.array_equal(img, last_img)

            if is_new_img:
                # Run inference!
                pred = self.predict(img, meta=meta)
                last_img = img


class VideoReader(threading.Thread):
    """Threaded video reader that yields frames at a fixed FPS.

    This class is useful for simulating a realtime feed.
    """

    daemon = True

    def __init__(
        self,
        video_path: str,
        fps: Optional[float] = None,
        grayscale: Optional[bool] = None,
    ):
        super().__init__()

        self.video_path = video_path
        self.fps = fps
        self.grayscale = grayscale
        self.vc = cv2.VideoCapture(video_path)

        if self.fps is None:
            self.fps = self.vc.get(cv2.CAP_PROP_FPS)

        if self.grayscale is None:
            _, img = self.vc.read()
            self.grayscale = self.check_grayscale(img)
            self.vc.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._num_frames = int(self.vc.get(cv2.CAP_PROP_FRAME_COUNT))
        self._height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))

        self._lock = threading.Lock()
        self._frame_idx = None
        self._image = None
        self._timestamp = None
        self.read_frame(0)

    @staticmethod
    def check_grayscale(img) -> bool:
        return (img[..., 0] == img[..., 1]).all()

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def __len__(self) -> int:
        return self.num_frames

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def channels(self) -> int:
        return 1 if self.grayscale else 3

    @property
    def shape(self):
        return (self.num_frames, self.height, self.width, self.channels)

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def frame_idx(self) -> int:
        with self._lock:
            return self._frame_idx

    @property
    def image(self) -> np.ndarray:
        with self._lock:
            return self._image

    @property
    def timestamp(self) -> float:
        with self._lock:
            return self._timestamp

    @property
    def last_data(self) -> Tuple[np.ndarray, Tuple[int, float]]:
        with self._lock:
            return self._image, (self._frame_idx, self._timestamp)

    def __str__(self):
        return f"VideoReader(video_path={self.video_path}, shape={self.shape}, fps={self.fps}, timestamp={self.timestamp}, frame_idx={self.frame_idx})"

    def read_frame(self, frame_idx: Optional[int] = None):
        if frame_idx is not None:
            self.vc.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        else:
            frame_idx = int(self.vc.get(cv2.CAP_PROP_POS_FRAMES))

        success, img = self.vc.read()

        if success:
            img = img.copy()
            if self.grayscale:
                img = img[:, :, [0]]
            else:
                img = img[:, :, ::-1]
            timestamp = perf_counter()
        else:
            img, frame_idx, timestamp = None, None, None

        with self._lock:
            self._image = img
            self._frame_idx = frame_idx
            self._timestamp = timestamp
        return img, frame_idx, timestamp

    def run(self):
        self.vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
        done = False
        timestamp = perf_counter()
        while not done:
            dt = perf_counter() - timestamp
            if dt >= self.dt:
                timestamp = perf_counter()
                img, frame_idx, timestamp_read = self.read_frame()
                if img is None:
                    done = True

import os
import numpy as np
import imageio.v2 as imageio


class VideoRecorder:

    def __init__(
            self,
            env,
            video_folder: str,
            video_interval: 100,
            video_length: int = 0,
            video_prefix: str = "rl-video",
            video_fps: int = 30,
    ):
        if env.render_mode in {None, "human", "ansi", "ansi_list"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with"
                f" RecordVideo. Initialize your environment with a render_mode"
                f" that returns an image, such as rgb_array."
            )
        self.env = env

        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)

        self.video_prefix = video_prefix
        self.video_length = video_length
        self.video_fps = video_fps
        self.video_interval = video_interval
        self.video_trigger = video_interval - video_length - 1

        self.video_id = 0
        self.clip = None
        self.recording = False
        self.recorded_frames = []

    def start_video_recorder(self, iter_id):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.recording = True
        self.video_id = iter_id

    def capture_frame(self):
        frame = self.env.unwrapped.render()
        self.recorded_frames.append(frame)

    def step(self, iter_id):
        if iter_id % self.video_interval == self.video_trigger and not self.recording:
            self.start_video_recorder(iter_id)

        if (iter_id - self.video_id) >= self.video_length and self.recording:
            self.close_video_recorder(iter_id)

        if self.recording:
            self.capture_frame()

    def close_video_recorder(self, iter_id):
        """Closes the video recorder if currently recording."""
        if iter_id is not None:
            numpy_to_video(np.array(self.recorded_frames).transpose(0, 1, 2, 3),
                           os.path.join(self.video_folder, f"{self.video_prefix}-step-{iter_id + 1}") + ".mp4",
                           framerate=self.video_fps)

        self.recording = False
        self.recorded_frames = []

    def close(self):
        self.close_video_recorder(iter_id=None)


def numpy_to_video(images: np.ndarray, output_file: str, framerate: int = 30):
    """
    Convert a NumPy array of shape (n, h, w, c) to a video file.

    Args:
        images: np.ndarray, shape (n, h, w, c), dtype uint8.
        output_file: str, path to the output .mp4 file.
        framerate: int, frames per second.
    """
    assert isinstance(images, np.ndarray), "Input must be a NumPy array"
    assert images.ndim == 4, "Input shape must be (n, h, w, c)"
    assert images.dtype == np.uint8, "Image array must be of type uint8"

    with imageio.get_writer(output_file, fps=framerate) as writer:
        for frame in images:
            writer.append_data(frame)

"""Fast GIF summaries for TensorBoard, encoded with ffmpeg.

tensorboardX's add_video converts to float32 and encodes through moviepy, which
is slow and an extra dependency. This pipes uint8 frames straight to ffmpeg
(palette-optimised GIF), adapted from dreamerv3's embodied logger.
"""

import shutil
import subprocess

import numpy as np


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def to_uint8(frames: np.ndarray) -> np.ndarray:
    if np.issubdtype(frames.dtype, np.floating):
        return np.clip(255 * frames, 0, 255).astype(np.uint8)
    return frames.astype(np.uint8, copy=False)


def make_grid(videos: np.ndarray) -> np.ndarray:
    """(B, T, H, W, C) -> (T, rows*H, cols*W, C), padding B to a power of two."""
    b, t, h, w, c = videos.shape
    if b & (b - 1):  # not a power of two
        pad = (1 << b.bit_length()) - b
        videos = np.concatenate(
            (videos, np.zeros((pad, t, h, w, c), dtype=videos.dtype)), axis=0
        )
    n = videos.shape[0]
    rows = 2 ** ((b.bit_length() - 1) // 2)
    cols = n // rows
    videos = videos.reshape(rows, cols, t, h, w, c)
    videos = videos.transpose(2, 0, 3, 1, 4, 5)
    return videos.reshape(t, rows * h, cols * w, c)


def encode_gif(frames: np.ndarray, fps: float) -> bytes:
    """(T, H, W, C) uint8, C in {1, 3} -> GIF bytes."""
    t, h, w, c = frames.shape
    pix_fmt = {1: "gray", 3: "rgb24"}[c]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-r", f"{fps:.02f}", "-s", f"{w}x{h}", "-pix_fmt", pix_fmt, "-i", "-",
        # no `fifo` between split and paletteuse: it was only a buffering
        # workaround for old ffmpeg, and ffmpeg 7 removed the filter
        "-filter_complex", "[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse",
        "-r", f"{fps:.02f}", "-f", "gif", "-",
    ]
    proc = subprocess.run(
        cmd, input=np.ascontiguousarray(frames).tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if proc.returncode:
        raise IOError(f"{' '.join(cmd)}\n{proc.stderr.decode('utf8', 'replace')}")
    return proc.stdout


def to_nthwc(vid: np.ndarray, dataformats: str) -> np.ndarray:
    """Any permutation of N, T, C, H, W (N optional) -> (N, T, H, W, C)."""
    dataformats = dataformats.upper()
    if "N" not in dataformats:
        vid = vid[None]
        dataformats = "N" + dataformats
    if sorted(dataformats) != sorted("NTCHW") or vid.ndim != 5:
        raise ValueError(f"dataformats {dataformats!r} does not match a {vid.ndim}-d video")
    return vid.transpose([dataformats.index(d) for d in "NTHWC"])


def video_summary(tag: str, vid: np.ndarray, fps: float, dataformats: str = "NTCHW"):
    from tensorboardX.proto.summary_pb2 import Summary

    frames = make_grid(to_uint8(to_nthwc(np.asarray(vid), dataformats)))
    t, h, w, c = frames.shape
    image = Summary.Image(height=h, width=w, colorspace=c, encoded_image_string=encode_gif(frames, fps))
    return Summary(value=[Summary.Value(tag=tag, image=image)])

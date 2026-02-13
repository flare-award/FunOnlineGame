import numpy as np

from autopilot_bot.capture import CaptureRegion, ScreenCapture


def test_preprocess_shape_color():
    cap = ScreenCapture(CaptureRegion(0, 0, 10, 10), image_size=(96, 96), grayscale=False)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    out = cap.preprocess(frame)
    assert out.shape == (96, 96, 3)
    assert out.dtype == np.float32


def test_preprocess_shape_gray():
    cap = ScreenCapture(CaptureRegion(0, 0, 10, 10), image_size=(64, 64), grayscale=True)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    out = cap.preprocess(frame)
    assert out.shape == (64, 64, 1)

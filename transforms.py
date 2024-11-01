import cv2
import numpy as np
from PIL import Image


class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)

class LineDrawingTransform:
    def __init__(self, threshold1=50, threshold2=150):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def __call__(self, img):
        # Convert the PIL Image to a NumPy array`
        img_np = np.array(img)

        # Convert to grayscale
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif len(img_np.shape) == 2 or img_np.shape[2] == 1:
            gray_image = img_np
        else:
            raise ValueError("unknown image shape:", img_np.shape)

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, self.threshold1, self.threshold2)

        # Invert the edges to get black on white line drawing
        line_drawing = cv2.bitwise_not(edges)

        # Convert the NumPy array back to a PIL Image
        return Image.fromarray(line_drawing)
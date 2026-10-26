"""Face-ID images: which cage triangle is visible at each pixel (spec §5.2).

Pure numpy, shared by the Blender renderer (encode) and the label stage (decode).
Each triangle index is stored as index + 1 in base 41 across R, G and B, one digit
per channel at byte value digit * 6 + 3. Steps of 6 mean colour-management
rounding of a byte or two cannot change the decoded index. Black (all digits 0)
is background, since index + 1 is never 0. Up to 41**3 - 1 = 68,920 triangles.
"""
import numpy as np

BASE = 41
STEP = 6
OFFSET = 3
LIMIT = BASE ** 3 - 1


def encode(count):
    """(count, 3) uint8 colours for triangles 0..count-1."""
    if count > LIMIT:
        raise ValueError(f'{count} triangles is more than the face-ID code can hold ({LIMIT})')
    value = np.arange(1, count + 1)
    digits = np.stack([value // BASE ** 2, (value // BASE) % BASE, value % BASE], axis=1)
    return (digits * STEP + OFFSET).astype(np.uint8)


def decode(rgb):
    """(H, W, 3) uint8 image -> (H, W) int64 triangle indices, -1 where nothing is visible."""
    digits = np.clip(np.rint((rgb[..., :3].astype(np.float64) - OFFSET) / STEP), 0, BASE - 1).astype(np.int64)
    value = digits[..., 0] * BASE ** 2 + digits[..., 1] * BASE + digits[..., 2]
    return value - 1

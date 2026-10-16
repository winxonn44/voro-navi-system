"""Q4: real segmentation on first-person video.

Conventions shared by every module here:
  frames   uint8 [H, W, 3] RGB numpy arrays
  masks    bool  [H, W], True = floor (free space) in the camera image
  BEV maps bool  [h, w] occupancy grids, True = obstacle (or unseen),
           the same convention as voronav.maps
"""

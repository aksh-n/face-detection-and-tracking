# face-detection-and-tracking
Detects face in the first frame of the video using Viola-Jones face detector, then tracks the face using mean-shift tracking.

- `mean_shift_hue` uses mean-shift tracking with the hue histograms of the face
- `mean_shift_gradient` uses mean-shift tracking with the gradient histograms of the grayscaled face

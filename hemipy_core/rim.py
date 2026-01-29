import numpy as np
import imageio.v3 as iio
from scipy.ndimage import label

# Load one image
p = r"Simulations\DHP - ERECT - 4000x4000\DHP - ERECT - 4000x4000\Case 001\Plot04\test45.png"
img = iio.imread(p)

print("Image shape:", img.shape)
print("Image dtype:", img.dtype)

# If RGB, convert to grayscale
if len(img.shape) == 3:
    img_gray = np.mean(img, axis=2)
else:
    img_gray = img

# Find non-zero region (black corners are 0 or very dark)
non_zero = img_gray > 10  # threshold to separate black from content

# Find connected component (the circular region)
labeled, num_features = label(non_zero)
print(f"Found {num_features} connected region(s)")

if num_features > 0:
    # Get the largest component (the circle)
    sizes = np.bincount(labeled.ravel())
    largest_label = np.argmax(sizes[1:]) + 1  # skip background (0)
    circle_mask = labeled == largest_label
    
    # Find the bounding circle: get all points in the circle
    y_coords, x_coords = np.where(circle_mask)
    
    # Optical centre (0-based indexing in array)
    opt_cen_0based = np.array([1999.5, 1999.5])  # equivalent to (2000.5, 2000.5) in 1-based
    
    # Distance from centre to each point
    distances = np.sqrt((y_coords - opt_cen_0based[0])**2 + (x_coords - opt_cen_0based[1])**2)
    
    # Max distance ≈ circle radius
    circle_radius = np.max(distances)
    
    print(f"\nCircle radius (pixels from centre to edge): {circle_radius:.1f}")
    print(f"Farthest point distance: {np.max(distances):.1f}")
    print(f"Nearest non-center point: {np.min(distances[distances > 0]):.1f}")
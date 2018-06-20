import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Identify obstacles (like colour thresh but opposite)
def obstacle_thresh(img, rgb_thresh=(160, 160, 160)):
	# Create an array of zeros same xy size as img, but single channel
	obstacle_select = np.zeros_like(img[:,:,0])

	# Opposite of color_thresh
	below_thresh = (img[:,:,0] < rgb_thresh[0]) \
				& (img[:,:,1] < rgb_thresh[1]) \
				& (img[:,:,2] < rgb_thresh[2])
	# Index the array of zeros with the boolean array and set to 1
	obstacle_select[below_thresh] = 1
	# Return the binary image
	return obstacle_select

# Identify pixels that look like rocks
""" This was not my idea, but I have lost the source and cannot properly credit them. """
def rock_thresh(img, low_rgb_thresh=(100, 100, 0), hi_rgb_thresh=(210, 210, 55)):
	# Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])

	# Threshold to only get yellow
	mask = cv2.inRange(img, low_rgb_thresh, hi_rgb_thresh)

	return mask

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    # mask added from project walkthrough
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))

    return warped, mask

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # NOTE: camera image is coming to you in Rover.img

    # Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 6

    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                    [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                    [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                    [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset]])

    # Apply perspective transform (mask added from project walkthrough)
    warped, _ = perspect_transform(Rover.img, source, destination)

    # Apply color threshold to identify navigable terrain/obstacles/rock samples
	obstacle = obstacle_thresh(warped)
	rock = rock_thresh(warped)
	navigable = color_thresh(warped)

    # See if there are obstacles (added from walkthrough video)
    # obstacles = np.absolute(np.float32(binary_image) - 1) * mask

    # Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obstacle * 255
	Rover.vision_image[:,:,1] = rock
    Rover.vision_image[:,:,2] = navigable * 255


    # Convert map image pixel values to rover-centric coords
	obst_xy = rover_coords(obstacle)
	rock_xy = rover_coords(rock)
	nav_xy = rover_coords(navigable)


    # Convert rover-centric pixel values to world coordinates
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw
    scale = 2 * dst_size

	obstacle_coord = pix_to_world(*obst_xy, xpos, ypos, yaw, \
                               Rover.worldmap.shape[0], scale)
	rock_coord = pix_to_world(*rock_xy, xpos, ypos, yaw, \
							   Rover.worldmap.shape[0], scale)
    navigable_coord = pix_to_world(*nav_xy, xpos, ypos, yaw, \
                               Rover.worldmap.shape[0], scale)

    # Update Rover worldmap (to be displayed on right side of screen)
	Rover.worldmap[obstacle_coord[1], obstacle_coord[0], 0] += 10
	Rover.woldmap[rock_coord[1], rock_coord[0], 1] += 10
	Rover.worldmap[navigable_coord[1], navigable_coord[0], 2] += 10

    # Convert rover-centric pixel positions to polar coordinates
	obst_polar = to_polar_coords(*obst_xy)
	rock_polar = to_polar_coords(*rock_xy)
	nav_polar = to_polar_coords(*nav_xy)

    # Update Rover pixel distances and angles
    Rover.nav_dists, Rover.nav_angles = nav_polar

	# Update Rover rock distances and angels
	Rover.rock_dists, Rover.rock_angles = rock_polar

    return Rover

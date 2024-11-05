import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import label
import numpy as np


# Image preprocessing: convert to grayscale, apply blurring, enhance contrast, and remove shadows
def preprocess_image(image):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise from the image
    image = cv2.GaussianBlur(image, (7, 7), 1.5)

    # Enhance image contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # Remove shadows from the image using a shadow mask
    minimal_tresh = 180
    _, shadow_mask = cv2.threshold(image, minimal_tresh, 255, cv2.THRESH_BINARY)
    image = cv2.bitwise_and(image, image, mask=shadow_mask)

    return image


# Detect edges in the image using the Prewitt operator
def detect_edges_with_prewitt(image):
    # Prewitt method
    prewitt_x = cv2.filter2D(image, -1, np.array([[1, 0, -1],
                                                  [1, 0, -1],
                                                  [1, 0, -1]]))
    prewitt_y = cv2.filter2D(image, -1, np.array([[1, 1, 1],
                                                  [0, 0, 0],
                                                  [-1, -1, -1]]))
    prewitt_x = prewitt_x.astype(np.float32)
    prewitt_y = prewitt_y.astype(np.float32)
    prewitt_combined = cv2.sqrt(prewitt_x ** 2 + prewitt_y ** 2)

    return prewitt_combined


# Group connected pixels into separate contours
def find_connected_components(contoured_img):
    # Find connected components
    labeled_array, num_features = label(contoured_img)

    # Group points for each contour
    grouped_contours = []
    for i in range(1, num_features + 1):
        # Get points that belong to one contour
        points = np.argwhere(labeled_array == i)
        grouped_contours.append([tuple(pt) for pt in points])

    return grouped_contours


# Simplify the contour using the Douglas-Peucker method by removing excess points
def simplify_contour_douglas_peucker(points, epsilon):
    points = np.array(points)  # Convert input data to a numpy array

    if len(points) < 2:
        return points

    # Find the maximum distance to the line defined by the first and last points
    start, end = points[0], points[-1]
    line_vec = end - start
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    # Calculate distances to the line for each point
    distances = np.array([np.linalg.norm(np.cross(line_vec, p - start)) / np.linalg.norm(line_vec) for p in points])

    max_dist = np.max(distances)
    index = np.argmax(distances)

    # If the maximum distance is greater than epsilon, split recursively
    if max_dist > epsilon:
        # Recursion for the subset of points before and after the furthest point
        left_part = simplify_contour_douglas_peucker(points[:index + 1], epsilon)
        right_part = simplify_contour_douglas_peucker(points[index:], epsilon)

        # Combine results, removing duplicate points
        return np.vstack((left_part[:-1], right_part))
    else:
        # If no significant points, return only the start and end points
        return np.array([start, end])


# Visualize contours on a plot
def visualize_contours(contours, title):
    plt.figure()
    for contour in contours:
        y_coords = [point[0] for point in contour]
        x_coords = [point[1] for point in contour]
        plt.plot(x_coords, y_coords, linestyle='-', color='blue', markersize=1)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis('equal')
    plt.show()


input_image = cv2.imread(f'photos/2.png')

# Display the original image
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.show()

# Preprocess the image
processed_image = preprocess_image(input_image)

# Display the processed image
plt.imshow(processed_image, cmap='gray')
plt.title('Preprocessed Image')
plt.axis('off')

plt.show()

# Find image contours using the Prewitt operator
contours = detect_edges_with_prewitt(processed_image)

# Display the detected contours
plt.imshow(contours, cmap='gray')
plt.title('Image with contours')
plt.axis('off')

plt.show()

# Convert contours to a binary array, with 100 as the threshold value
contours = contours > 100

# Extract separate contours
contours = find_connected_components(contours)

visualize_contours(contours, 'Separated contours')

# Reduce the number of points in the contours using the Douglas-Peucker method
simplified_contours = []
for contour in contours:
    simplified_contour = simplify_contour_douglas_peucker(contour, epsilon=6)

    simplified_contours.append(simplified_contour)

visualize_contours(simplified_contours, 'Simplified contours')

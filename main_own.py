import cv2
import numpy as np

path_ball = "C:/A_Project_MS/PA1_dataset/chromeball/"
path = "C:/A_Project_MS/PA1_dataset/moai/"
#1~12.png in filename.txt
#x y z in light_directions.txt
#A B C in light_intensities.txt

mask_ball = cv2.imread(path_ball+"mask.JPG",cv2.IMREAD_GRAYSCALE)
mask_ball = cv2.resize(mask_ball, (0, 0), fx=0.5, fy=0.5) 

height, width = mask_ball.shape
center_x, center_y = width // 2, height // 2
crop_size=500
crop_mask = mask_ball[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size]

circles = cv2.HoughCircles(crop_mask, cv2.HOUGH_GRADIENT, 1, 50, param1 = 100, param2 = 40, minRadius = 200, maxRadius = 400)
'''
for i in circles[0]:
    cv2.circle(crop_mask, (int(i[0]), int(i[1])), int(i[2]), (0,0,255), 2)
cv2.imshow("circle", crop_mask)
cv2.waitKey(0)
'''
circle_x, circle_y, circle_r = center_x - crop_size + circles[0][0][0], center_y - crop_size + circles[0][0][1], circles[0][0][2]

images_ball = []
for num in range(1,13):
    img = cv2.imread(path_ball+str(num)+".JPG",cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5) 
    images_ball.append(img)
'''
cv2.imshow("Test Image", images_ball[11])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

num_images = 12
light_directions = []
for i in range(num_images):
    blurred = cv2.GaussianBlur(images_ball[i], (55, 55), 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    '''
    cv2.circle(images_ball[i], max_loc, 10, (0, 255, 0), 4) 
    cv2.imshow("circle", images_ball[i])
    cv2.waitKey(0)
    '''
    z = (circle_r ** 2 - ((center_x - max_loc[0])** 2 + (center_y - max_loc[1])** 2) ) **0.5
    light_directions.append([max_loc[0] - center_x, center_y - max_loc[1], z])

############################################################################################################
images = []
for num in range(1,13):
    img = cv2.imread(path+str(num)+".JPG",cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5) 
    images.append(img)

mask = cv2.imread(path+"mask.JPG",cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)
mask = cv2.GaussianBlur(mask, (55, 55), 0)
mask[mask >= 100] = 255
mask[mask < 100] = 0

cv2.imshow("Test Image", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

pixel_intencity = [[] for _ in range(num_images)]
pixel_info = []
num_pixels = 0

for i in range(num_images):
    for h in range(height):
        for w in range(width):
            if mask[h,w] == 255:
                pixel_intencity[i].append(images[i][h,w])
                if i == 0:
                    pixel_info.append([h,w])
                    num_pixels += 1

D = np.zeros((num_pixels, num_images), dtype=np.float32)

for i in range(num_images):
    D[:, i] = pixel_intencity[i]
light_directions_inv = np.transpose(np.linalg.pinv(light_directions))
normal = D @ light_directions_inv
norms = np.linalg.norm(normal, axis=1, keepdims=True)
normalized_normal = normal / norms

normal_image = np.zeros((height, width,3), dtype=np.float32)
for idx, (h, w) in enumerate(pixel_info):
    Nx, Ny, Nz = normalized_normal[idx]
    normal_image[h, w, 2] = (Nx + 1) * 0.5 * 255
    normal_image[h, w, 1] = (Ny + 1) * 0.5 * 255
    normal_image[h, w, 0] = (Nz + 1) * 0.5 * 255

image = np.clip(normal_image, 0, 255).astype(np.uint8)

cv2.imshow("Normal Map", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
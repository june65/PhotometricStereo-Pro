import cv2
import numpy as np

unknown_img = cv2.imread("unknown.JPG",cv2.IMREAD_GRAYSCALE)
unknown_img = cv2.resize(unknown_img, (0, 0), fx=0.5, fy=0.5)
height, width = unknown_img.shape

#light_directions
light_directions = np.load('light_directions.npy')
#albedos
albedos = np.load('albedos.npy')
#normalized_normal
normalized_normal = np.load('normalized_normal.npy')
#pixel_info
pixel_info = np.load('pixel_info.npy')

#############relighting###############
GT_light_directions = light_directions[11]
norm_light = np.linalg.norm(light_directions[11])
GT_light_directions = GT_light_directions / norm_light
#print(light_directions)

pixel_intencity = []
for idx, (h, w) in enumerate(pixel_info):
    pixel_intencity.append(unknown_img[h,w])
albedos_f = albedos.flatten()
normalized_normal_inv = np.transpose(np.linalg.pinv(normalized_normal))

unknown_light_directions = (pixel_intencity / albedos_f) @ normalized_normal_inv
norm_light = np.linalg.norm(unknown_light_directions)
unknown_light_directions = unknown_light_directions / norm_light

albedos_t = np.transpose(albedos)
unknown_light_directions_t = np.transpose(GT_light_directions)
result = albedos_t * (normalized_normal @ unknown_light_directions_t)
result_t = np.transpose(result)

result_image = np.zeros((height, width,1), dtype=np.float32)
for idx, (h, w) in enumerate(pixel_info):
    result_image[h, w] = result_t[idx]

cv2.imshow("Result Image", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#############mseloss###############
mse_error = 0
for idx, (h, w) in enumerate(pixel_info):
    mse_error += (unknown_img[h, w]/255 - result_image[h, w]) ** 2
mse_error = (mse_error / (idx+1))[0]
print(f"mse_error:{mse_error}")

cos_theta = np.dot(unknown_light_directions, GT_light_directions) / (np.linalg.norm(unknown_light_directions) * np.linalg.norm(GT_light_directions))
cos_theta = np.clip(cos_theta, -1.0, 1.0) 
theta = np.degrees(np.arccos(cos_theta))

light_error = np.sqrt(np.mean((unknown_light_directions - GT_light_directions) ** 2) / 3)
print(f"error_degree:{theta}")

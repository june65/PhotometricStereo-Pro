import cv2
import numpy as np

path = "C:/A_Project_MS/DiLiGenT_Test/DiLiGenT_Test/pmsData/bearPNG/"
#001~096.png in filename.txt
#x y z in light_directions.txt
#A B C in light_intensities.txt

with open(path + "filenames.txt", "r", encoding="utf-8") as f:
    imagepaths = [line.strip() for line in f]

images = []
for filename in imagepaths:
    img = cv2.imread(path+filename,cv2.IMREAD_GRAYSCALE)
    images.append(img)

num_images = len(images)

'''
cv2.imshow("Test Image", images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

with open(path + "light_directions.txt", "r", encoding="utf-8") as f:
    light_directions = [list(map(float, line.strip().split(" "))) for line in f]
    
with open(path + "light_intensities.txt", "r", encoding="utf-8") as f:
    light_intensities = [list(map(float,line.strip().split(" "))) for line in f]

mask = cv2.imread(path+"mask.png",cv2.IMREAD_GRAYSCALE)
height, width = mask.shape

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
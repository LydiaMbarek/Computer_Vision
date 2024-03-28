import cv2
import numpy as np
OPEN = "OPEN"
CLOSE  = "CLOSE"
def trouver_contours(image,seuil_contour=0.5):
    contours_image = np.zeros_like(image).astype(np.uint8)
    points = []
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            diff_x = int(image[y, x + 1]) - int(image[y, x - 1])
            diff_y = int(image[y + 1, x]) - int(image[y - 1, x])
            gradient_magnitude = np.sqrt(diff_x**2 + diff_y**2)
            if gradient_magnitude > seuil_contour:
                contours_image[y, x] = 255
                points.append((y,x))
    return contours_image, points
def blur(img, size_kernel):
    img_filter = []
    height, width = img.shape

    for i in range(width):
        row = []
        for j in range(height):
            avg = 0
            x0 = int(i - size_kernel / 2)
            y0 = int(j - size_kernel / 2)
            sum_kernel = 0  # Variable to store the sum of kernel elements
            for k1 in range(size_kernel):
                for k2 in range(size_kernel):
                    if x0 + k1 >= 0 and y0 + k2 >= 0 and x0 + k1 < width and y0 + k2 < height:
                        avg += img[y0 + k2][x0 + k1]
                        sum_kernel += 1  # Increment the sum for each valid pixel
            avg = avg / sum_kernel  # Normalize by the sum of kernel elements
            row.append(int(avg))
        img_filter.append(row)

    return np.array(img_filter)

def blur_3d(img,size_kernel):
    weight , height , _ = img.shape
    img_output = []
    for i in range(3):
        x = blur(img[:,:,i].astype(np.uint8),size_kernel)
        img_output.append(x)
    out_put = np.array(img_output).T
    return out_put.astype(np.uint8)

def inRange(image,low,high):
    height , weight , _ = image.shape
    new_image = np.zeros((height,weight))
    i = 0
    while(i<height):
        j=0
        while(j<weight):
            if low[0] <= image[i][j][0] <= high[0] and low[1] <= image[i][j][1] <= high[1]  and low[2] <= image[i][j][2] <= high[2]  : 
                new_image[i][j] = 1
            j += 1           
        i += 1
    new_image = new_image*255
    return new_image

def erode(img, kernel=[[1, 1, 1], [1, 1, 1], [1, 1, 1]]):
    img = img // 255
    img_filter = np.ones_like(img)
    size_kernel = len(kernel)
    height, weight = img.shape
    for i in range(height):
        for j in range(weight):
            x0 = i - size_kernel // 2
            y0 = j - size_kernel // 2
            for k1 in range(size_kernel):
                for k2 in range(size_kernel):
                    if x0 + k1 >= 0 and y0 + k2 >= 0 and x0 + k1 < height and y0 + k2 < weight:
                        if kernel[k1][k2] != img[x0 + k1, y0 + k2]:
                            img_filter[i, j] = 0
                            break
                if img_filter[i, j] == 0:
                    break

    return np.array(img_filter) * 255

def dilate(img,kernel=[[1,1,1],[1,1,1],[1,1,1]]):
    img = img // 255
    img_filter = np.zeros_like(img)
    size_kernel = len(kernel)
    height, weight = img.shape
    for i in range(height):
        for j in range(weight):
            x0 = i - size_kernel // 2
            y0 = j - size_kernel // 2
            for k1 in range(size_kernel):
                for k2 in range(size_kernel):
                    if x0 + k1 >= 0 and y0 + k2 >= 0 and x0 + k1 < height and y0 + k2 < weight:
                        if kernel[k1][k2] == 1 and img[x0 + k1, y0 + k2] == 1:
                            img_filter[i, j] = 1
                            break
                if img_filter[i, j] == 1:
                    break
    return np.array(img_filter) * 255

def morphEx(img,size,type):
    kernel = np.ones((size,size))
    if type == 'OPEN':
        img_erosion = erode(img,kernel)
        img_morphEx = dilate(img_erosion,kernel)
    else :
        img_dilate = dilate(img,kernel)
        img_morphEx = erode(img_dilate,kernel)
    return img_morphEx
def GreenScreen(imageSource, imageBackGround,low,high):
    image_copy = cv2.cvtColor(imageSource, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_copy, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
    weigt , high = mask.shape
    image_OutPut = np.array(imageBackGround)
    for x in range(weigt):
        for y in range(high):
            if mask[x,y] == 255:
                image_OutPut[x,y] = imageSource[x,y]
    return image_OutPut.astype(np.uint8)

def detect_image(image,low,high):
    image_copy = image.copy()
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_copy,low,high)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, None, iterations=2)
    #elements = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #elements = sorted(elements, key=lambda x:cv2.contourArea(x), reverse=True)
    elements, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # List to store points
    points = []
    # Iterate through contours
    for element in elements:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(element)
        # Iterate through pixels within the bounding box
        for pixel_y in range(y, y + h):
            for pixel_x in range(x, x + w):
                points.append((pixel_x, pixel_y))
    return image_copy,mask,points

def invisibility_cloak(img,low,high):
    image_copy = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_copy, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
    dict = {}
    x, y = mask.shape
    for i in range(x):
        for j in range(y):
            if mask[i,j] == 0 : continue
            if i not in dict: dict[i] = []
            dict[i].append(j)
    for y in dict.keys():
        pas = [0,0,0]
        pred = [0,0,0]
        xs = dict[y]
        if xs[0] > 0 :
            for i in range(3):
                pas[i] = pas[i] + img[y , xs[0] - 1 , i]
                pred[i] = img[y , xs[0] - 1,i]
        if xs[-1] < (img.shape[1] - 1) :
            for i in range(3):
                pas[i] = img[y , xs[-1] + 1 , i] - pas[i]
                pas[i] = pas[i] / len(xs)
        for i in range(len(xs)):
            for j in range(3):
                img[y,xs[i],j] = pred[j] + pas[j]
                pred[j] = img[y,xs[i],j]
    return img.astype(np.uint8)


def filtreMedian(img,vois):  # chaque pixel remplace par la med de ses voisinages
    h, w = img.shape
    imgMed = np.zeros(img.shape, img.dtype)
    for y in range(h):
        for x in range(w):
            if (y < vois/2 or y > h-vois/2 or x < vois/2 or y > w-vois/2):
                imgMed[y, x] = img[y, x]
            else:
                imgV = img[int(y - vois/2): int(y + vois/2), int(x - vois/2): int(x + vois/2)]
                t = np.zeros((vois*vois), np.uint8)
                #
                for yv in range(imgV.shape[0]):
                    for xv in range(imgV.shape[1]):
                        t[yv*vois+xv] = imgV[yv, xv]
                t.sort()
                imgMed[y, x] = t[int((vois*vois)/2)+1]
    return imgMed

def gaussian_kernel(size, sigma=1.0):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def apply_gaussian_filter(matrix, kernel_size=3, sigma=1.0):
    kernel = gaussian_kernel(kernel_size, sigma)
    height = len(matrix)
    width = len(matrix[0])
    filtered_matrix = [[0 for _ in range(width)] for _ in range(height)]
    kernel_size = len(kernel)
    k = kernel_size // 2
    for i in range(height):
        for j in range(width):
            sum_val = 0
            for ki in range(-k, k + 1):
                for kj in range(-k, k + 1):
                    if 0 <= i + ki < height and 0 <= j + kj < width:
                        sum_val += matrix[i + ki][j + kj] * \
                            kernel[ki + k][kj + k]
            filtered_matrix[i][j] = sum_val

    return np.uint8(filtered_matrix)
def apply_laplacian_filter(matrix):
    kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    height,width = matrix.shape
    filtered_matrix = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(1, height-1):
        for j in range(1, width-1):
            sum_val = 0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    sum_val += matrix[i + ki,j + kj] * kernel[ki + 1][kj + 1]
            filtered_matrix[i][j] = sum_val

    return np.uint8(filtered_matrix)


def trouver_contours_center(image,seuil_contour=0.5):
    contours_image = np.zeros_like(image).astype(np.uint8)
    points = []
    minx , maxx = None , None
    miny , maxy = None , None
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            diff_x = int(image[y, x + 1]) - int(image[y, x - 1])
            diff_y = int(image[y + 1, x]) - int(image[y - 1, x])
            gradient_magnitude = np.sqrt(diff_x**2 + diff_y**2)
            if gradient_magnitude > seuil_contour:
                contours_image[y, x] = 255
                points.append((y,x))
                if minx == None :
                    minx = x
                if miny == None :
                    miny = y
                elif y < miny : miny = y
                maxx = x
                if maxy == None:
                    maxy = y
                elif y > maxy : maxy = y
    center = None
    if minx!= None :
        y_center = (maxy + miny) //2
        x_center = (maxx + minx) // 2
        center = (y_center,x_center)
    return contours_image, points , center

def detect_Image(image, low, high):
    points = []
    image_operate = image.copy()
    image_operate = cv2.cvtColor(image_operate, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_operate, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
    contours_image, points, center = trouver_contours_center(mask)
    if center != None:
        center_x, center_y = center[0], center[1]
        cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), 2)
    return image
def detect_Center(image, low, high):
    points = []
    image_operate = image.copy()
    image_operate = cv2.cvtColor(image_operate, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_operate, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
    cv2.imshow("image",image)
    cv2.imshow("mask",mask)
    contours_image, points, center = trouver_contours_center(mask)
    if center != None:
        center_x, center_y = center[0], center[1]
        cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), 2)
    return center
def bilateral(src, kernel_size, sigma_s, sigma_r):
    """
    Apply bilateral filter to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel_size (Sequence[int]): The kernel size.
        sigma_s (float): The standard deviation of the spatial gaussian distribution.
        sigma_r (float): The standard deviation of the range gaussian distribution.

    Returns:
        numpy.ndarray: The filtered image.
    """
    if not isinstance(kernel_size, tuple):
        raise ValueError("kernel_size should be a tuple")
    if len(kernel_size) != 2:
        raise ValueError("kernel_size should be a tuple of length 2")
    if not isinstance(src, np.ndarray):
        raise ValueError("src should be a numpy array")
    if len(src.shape) != 2:
        raise ValueError("src should be a 2D array")

    if sigma_s <= 0:
        sigma_s = src.std()
    if sigma_r <= 0:
        sigma_r = src.std()

    dst = np.zeros_like(src)
    pad_width = ((kernel_size[0] // 2, kernel_size[0] // 2), (kernel_size[1] // 2, kernel_size[1] // 2))
    padded_src = np.pad(src, pad_width, 'edge')

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            kernel = np.zeros(kernel_size)
            for k in range(kernel_size[0]):
                for l in range(kernel_size[1]):
                    kernel[k, l] = np.exp(-((k - kernel_size[0] // 2) ** 2 + (l - kernel_size[1] // 2) ** 2) / (
                            2 * sigma_s ** 2)) * np.exp(
                        -((padded_src[i, j] - padded_src[i + k, j + l]) ** 2) / (2 * sigma_r ** 2))
            kernel /= np.sum(kernel)
            dst[i, j] = np.sum(kernel * padded_src[i:i + kernel_size[0], j:j + kernel_size[1]])

    return dst
def apply_kernel(image):
    kernel = [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ]
    kernel_size = len(kernel)
    height = len(image)
    width = len(image[0])
    output = [[0 for _ in range(width)] for _ in range(height)]

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            sum_val = 0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    sum_val += kernel[ki][kj] * image[i + ki - 1][j + kj - 1]
            output[i][j] = sum_val

    return output
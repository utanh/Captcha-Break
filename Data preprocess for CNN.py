import cv2
from sklearn.cluster import KMeans
from pathlib import Path
import numpy as np
import random

def findOutlier_morePixel(pixel_values, pixel_id):
  kmeans = KMeans(n_clusters=2)
  kmeans.fit(pixel_values)

  label_is_outlier = 0
  num_zero_label = np.count_nonzero(kmeans.labels_ == np.array(0))

  if num_zero_label > len(pixel_values) - num_zero_label:
    label_is_outlier = 0
  else:
    label_is_outlier = 1

  outlier_pixels_id = []
  for i in range(len(pixel_id)):
    if kmeans.labels_[i] == label_is_outlier:
      outlier_pixels_id.append(pixel_id[i])
      
  return outlier_pixels_id

def findChar_Pixel(pixel_values, pixel_id):
  kmeans = KMeans(n_clusters=5)
  kmeans.fit(pixel_values)
 
  unique_ = np.unique(kmeans.labels_, return_counts=True)
  label_is_char = unique_[0][np.argmax(unique_[1])]
  char_pixels_id = []
  for i in range(len(pixel_id)):
    if kmeans.labels_[i] == label_is_char:
      char_pixels_id.append(pixel_id[i])

  return char_pixels_id

def padding(img):
  new_square = 0
  if img.shape[0] > img.shape[1]:
    new_square = img.shape[0]
  else:
    new_square = img.shape[1]

  color = (0,0,0)
  result = np.full((new_square, new_square, 3), color, dtype=np.uint8)

  # compute center offset
  x_center = (new_square - img.shape[1]) // 2
  y_center = (new_square - img.shape[0]) // 2

  result[y_center : y_center + img.shape[0], x_center : x_center + img.shape[1]] = img
  return result

def finhLocat_Captcha(img):
  red = img[:,:,0] // 255
  sum_row = [sum(red[i]) for i in range(len(red))]
  row_compare0 = [i for i in range(len(sum_row)) if sum_row[i] > 4]
  bottom_idx = max(row_compare0)
  top_idx = min(row_compare0)

  sum_column = [sum(red[:,i]) for i in range(red.shape[1])]
  column_greater0 = [i for i in range(len(sum_column)) if sum_column[i] > 4 ]
  right_idx = max(column_greater0)
  left_idx = min(column_greater0)

  return [top_idx, left_idx, bottom_idx, right_idx]

path_soure = 'C:/Users/anhtu/OneDrive/Desktop/5k_captcha_data/'
path_dict = 'C:/Users/anhtu/OneDrive/Desktop/captcha_dataset/'
data_dir = Path(path_soure)
# List path of image
img_paths = list(map(str, list(data_dir.glob("*.png"))))

random_list = []
for i in range(len(img_paths)):
    n = random.randint(0, len(img_paths)-1)
    if n not in random_list:
        random_list.append(n)

for i in random_list:
    img = cv2.imread(img_paths[i])
    label_path = img_paths[i].split('\\')[-1]
    cv2.imwrite(path_dict + label_path, img)
    img = cv2.resize(img, (128,128), interpolation= cv2.INTER_LINEAR)

    pixel_poss = []
    pixel_values = []
    pixel_id = list(range(img.shape[0]*img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_poss.append([i, j])
            pixel_values.append(img[i][j])
    pixel_poss = np.array(pixel_poss)
    pixel_values = np.array(pixel_values)

    # Remove outlier in img
    outlier_pixels_id = []
    outlier_pixels_id = outlier_pixels_id + findOutlier_morePixel(pixel_values, pixel_id)

    pixel_id_new = [id for id in pixel_id if id not in outlier_pixels_id]
    pixel_poss_new = [pixel_poss[id] for id in pixel_id_new]
    pixel_values_new = [pixel_values[id] for id in pixel_id_new]

    char_pixels_id = findChar_Pixel(pixel_values_new, pixel_id_new)

    # Get remove outlier img
    for id in char_pixels_id:
      img[pixel_poss[id][0]][pixel_poss[id][1]] = np.array([255,255,255])

    for i in range(len(pixel_poss)):
        if False not in (img[pixel_poss[i][0]][pixel_poss[i][1]] == np.array([255,255,255])):
            continue
        img[pixel_poss[i][0]][pixel_poss[i][1]] = np.array([0,0,0])

    # Preproces img
    # columns = [img[:,i] for i in range(img.shape[1])]
    # num_pixel_is_c_in_column = [np.count_nonzero(columns[i] == np.array([255,255,255]))/3 for i in range(len(columns))]
    # maybe_character_columns = np.where(np.array(num_pixel_is_c_in_column) / img.shape[0] > 0.01)[0]

    # rows = [img[i,:,:] for i in range(img.shape[0])]
    # num_pixel_is_c_in_row = [np.count_nonzero(rows[i] == np.array([255,255,255]))/3 for i in range(len(rows))]
    # maybe_character_rows = np.where(np.array(num_pixel_is_c_in_row) / img.shape[1] > 0.01)[0]

    locat = finhLocat_Captcha(img)
    cropped_img = img[locat[0]:locat[2], locat[1]:locat[3], :]
    cv2.imwrite(path_dict + label_path.replace(".png", "") + '_cropped.png', cropped_img)

    kernel = np.array([[0,1,0], 
                       [1,1,0], 
                       [0,0,0]])
    morepixel_img = cv2.filter2D(cropped_img, -1, kernel)
    cv2.imwrite(path_dict + label_path.replace(".png", "") + '_morepixel.png', morepixel_img)

    padded_img = padding(morepixel_img)
    cv2.imwrite(path_dict + label_path.replace(".png", "") + '_padded.png', padded_img)

    #resized_img = cv2.resize(padded_img, (64,64), interpolation= cv2.INTER_LINEAR)
    #cv2.imwrite(path_dict + label_path.replace(".png", "") + '_preprocessed.png', resized_img)

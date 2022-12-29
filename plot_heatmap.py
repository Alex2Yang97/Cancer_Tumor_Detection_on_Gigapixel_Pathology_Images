# /**
#  * @author Zhirui(Alex) Yang
#  * @email zy2494@columbia.edu
#  * @create date 2022-12-27 23:38:19
#  * @modify date 2022-12-27 23:38:19
#  * @desc [description]
#  */


import numpy as np
from processing import get_center, delete_gray, get_target, get_patch_from_center, get_patches_from_center



def label_prediction(x, y, predict_mask, label=1.0, patch_len=299, target_size=128):
  offset = (patch_len - target_size)//2
  x_target = x + offset
  y_target = y + offset

  target_area = predict_mask[x_target: x_target+target_size, y_target: y_target+target_size]
  assert target_area.shape[0] == target_size

  predict_mask[x_target: x_target+target_size, y_target: y_target+target_size] = label

  return predict_mask


def label_heatmap_onezoom(
    model, slide_image, mask_image, predict_mask,
    level=5, threshold=0.5, step=10, patch_len=299, 
    show_res=True, verbose=0):

  x_scan = 0
  while x_scan <= slide_image.shape[0] - patch_len:
    
    y_scan = 0
    while y_scan <= slide_image.shape[1] - patch_len:
      
      xc_0, yc_0 = get_center(slide_image, x_scan, y_scan, level=level)

      factor = 2 ** level
      xc_level = xc_0 // factor
      yc_level = yc_0 // factor
      x_level = xc_level - patch_len//2
      y_level = yc_level - patch_len//2

      assert 0 <= x_level + patch_len-1 <= slide_image.shape[0]
      assert 0 <= y_level + patch_len-1 <= slide_image.shape[1]
      assert 0 <= x_level <= slide_image.shape[0]
      assert 0 <= y_level <= slide_image.shape[1]

      tumor_region = get_patch_from_center(slide_image, xc_0, yc_0, level=level)
      
      if delete_gray(tumor_region):
        check_image = get_patch_from_center(mask_image, xc_0, yc_0, level=level)
        label = get_target(check_image)

        tumor_region = tumor_region / 255.0
        
        pred = model.predict(np.array([tumor_region]), verbose=verbose)
        pred_label = (pred > threshold).astype("int32")

        if show_res:
          check_image = get_patch_from_center(mask_image, xc_0, yc_0, level=level)
          label = get_target(check_image)
          if label == 1:
            print(label, pred)

        if pred_label[0][0] == 1:
          predict_mask = label_prediction(x_scan, y_scan, predict_mask, label=1)
        else:
          predict_mask = label_prediction(x_level, y_level, predict_mask, label=0)

      y_scan += step
    x_scan += step
  
  return predict_mask


def label_heatmap_multizooms(
    model, multi_slide_images, multi_mask_images, predict_mask, 
    threshold=0.5, step=10, pred_level=5, patch_len=299, 
    show_res=True, verbose=0):

  x_scan = 0
  while x_scan <= multi_slide_images[-1].shape[0] - patch_len:
    
    y_scan = 0
    while y_scan <= multi_slide_images[-1].shape[1] - patch_len:
      
      xc_0, yc_0 = get_center(slide_image, x_scan, y_scan, level=level)
      
      factor = 2 ** pred_level
      xc_level = xc_0 // factor
      yc_level = yc_0 // factor
      x_level = xc_level - patch_len//2
      y_level = yc_level - patch_len//2

      assert 0 <= x_level + patch_len-1 <= multi_slide_images[0].shape[0]
      assert 0 <= y_level + patch_len-1 <= multi_slide_images[0].shape[1]
      assert 0 <= x_level <= multi_slide_images[0].shape[0]
      assert 0 <= y_level <= multi_slide_images[0].shape[1]

      multi_tumor_regions = get_patches_from_center(multi_slide_images, xc_0, yc_0, level_lst=level_lst)
      
      if delete_gray(multi_tumor_regions[0]):
        multi_tumor_regions = [region / 255.0 for region in multi_tumor_regions]

        input_data = {
            "input1": np.array([multi_tumor_regions[0]]), 
            "input2": np.array([multi_tumor_regions[1]]),
            "input3": np.array([multi_tumor_regions[2]])
            }
        pred = model.predict(input_data, verbose=verbose)
        pred_label = (pred > threshold).astype("int32")

        if show_res:
          multi_check_images = get_patches_from_center(multi_mask_images, xc_0, yc_0, level_lst=level_lst)
          label = get_target(multi_check_images[0])
          if label == 1:
            print(label, pred)

        if pred_label[0][0] == 1:
          predict_mask = label_prediction(x_level, y_level, predict_mask, label=1)
        else:
          predict_mask = label_prediction(x_level, y_level, predict_mask, label=0)

      y_scan += step
    x_scan += step

  return predict_mask

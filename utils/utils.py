import cv2
def resize_pad(image, target_shape):
    scale = min(target_shape[0]/image.shape[0],target_shape[1]/image.shape[1])
    image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
    hdif =  target_shape[0] - image.shape[0]
    wdif =  target_shape[1] - image.shape[1]
    buttom, top, right, left = hdif//2, hdif//2, wdif//2, wdif//2
    if hdif%2 != 0: top +=1
    if wdif%2 != 0: left+=1
    return cv2.copyMakeBorder(
        image, top, buttom, left, right, cv2.BORDER_CONSTANT,
        value=[0,0,0])
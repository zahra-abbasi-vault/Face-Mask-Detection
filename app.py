import cv2
from keras.models import load_model
from utils.utils import resize_pad
import numpy as np

cascPath = r"utils\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def faceDetection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.MOTION_TRANSLATION #cv2.CASCADE_DO_ROUGH_SEARCH #cv2.CASCADE_DO_ROUGH_SEARCH #cv2.CASCADE_SCALE_IMAGE
    )
    return faces


def main():

    input_shape  = (100,100,1)
    model_path = r"model_files\models"
    model = load_model(model_path) 
    model.summary()
    classes = sorted(["with_mask", "without_mask"])

    
    cap = cv2.VideoCapture(0)
    while True:        
        ret, image = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        if image is None: continue
        offset = 15
        faces = faceDetection(image)
        for face in faces:
            x, y, w, h = face
            y0_crop = max(0,y-offset)
            y1_crop = min(image.shape[0],y+h+offset)
            x0_crop = max(0,x-offset)
            x1_crop = min(image.shape[1],x+w+offset)
            image_faces = image[y0_crop:y1_crop, x0_crop:x1_crop]
            resized_image = resize_pad(image_faces, input_shape[:2])
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)/255
            prepared_image = np.expand_dims(gray_image, axis=(0, 3))

            output = model.predict(prepared_image, verbose=0)[0]
            output_class = classes[np.argmax(output)]
            print(output,output_class)
            if output_class =="with_mask":
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)
            cv2.rectangle(image, (x0_crop, y0_crop), (x1_crop, y1_crop), color, 2)
        cv2.imshow("main image", image)
        cv2.waitKey(20)
        # cv2.destroyAllWindows()

if __name__=="__main__": main()


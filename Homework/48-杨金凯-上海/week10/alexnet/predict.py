import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

K.set_image_dim_ordering('tf')

if __name__ == "__main__":
    model = AlexNet()
    # model.load_weights("./logs/ep039-loss0.004-val_loss0.652.h5")
    model.load_weights("./logs/ep036-loss0.000-val_loss0.831.h5")
    img = cv2.imread("./test_image/Test.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor,axis = 0)  # 维度扩展 在第0维上加一个维度  例如[[1,2],[2,3]]-->[[[1,2],[2,3]]]
    img_resize = utils.resize_image(img_nor,(224,224))
    #utils.print_answer(np.argmax(model.predict(img)))
    print(utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("ooo",img)
    cv2.waitKey(0)
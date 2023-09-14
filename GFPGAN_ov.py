import cv2
from openvino.runtime import Core  ###pip install openvino-dev[ONNX]==2023.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
import numpy as np

class GFPGAN:
    def __init__(self, model_path="GFPGANv1.4.onnx", device='cpu'):
        ie = Core()
        model = ie.read_model(model_path)
        self.net = ie.compile_model(model=model, device_name=device.upper())
        _, _, self.input_height, self.input_width = tuple(self.net.inputs[0].shape)

    def preprocess(self, img):
        img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)[:,:,::-1] / 255.0
        img = img.transpose((2, 0, 1))
        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def postprocess(self, img):
        img = (img.transpose(1,2,0).clip(-1,1) + 1) * 0.5
        img = (img * 255)[:,:,::-1]
        img = img.clip(0, 255).astype('uint8')
        return img

    def enhance(self, img):
        img = self.preprocess(img)
        output = self.net(img)[0][0]
        output = self.postprocess(output)
        return output
    
if __name__=='__main__':
    mynet = GFPGAN(model_path="weights/GFPGANv1.4.onnx", device="cpu")
    image_path = 'test_images/1.jpg'
    img = cv2.imread(image_path)

    dstimg = mynet.enhance(img)
    cv2.namedWindow('srcimg', cv2.WINDOW_NORMAL)
    cv2.imshow('srcimg', img)
    winName = 'GFPGAN in OpenVINO'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

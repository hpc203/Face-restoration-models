import cv2
import onnxruntime
import numpy as np

class CodeFormer:
    def __init__(self, model_path="codeformer.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        self.resolution = self.session.get_inputs()[0].shape[-2:]

    def preprocess(self, img, w):
        img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)[:,:,::-1] / 255.0
        img = img.transpose((2, 0, 1))
        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, axis=0).astype(np.float32)
        w = np.array([w], dtype=np.double)
        return img, w

    def postprocess(self, img):
        img = (img.transpose(1,2,0).clip(-1,1) + 1) * 0.5
        img = (img * 255)[:,:,::-1]
        img = img.clip(0, 255).astype('uint8')
        return img

    def enhance(self, img, w=0.9):
        img, w = self.preprocess(img, w)
        output = self.session.run(None, {'x':img, 'w':w})[0][0]
        output = self.postprocess(output)
        return output

if __name__=='__main__':
    mynet = CodeFormer(model_path="weights/codeformer.onnx", device="cpu")
    image_path = 'test_images/1.jpg'
    img = cv2.imread(image_path)

    dstimg = mynet.enhance(img)
    cv2.namedWindow('srcimg', cv2.WINDOW_NORMAL)
    cv2.imshow('srcimg', img)
    winName = 'CodeFormer in OnnxRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
from detectron2.engine import DefaultPredictor
import torch
import torch.nn as nn
import numpy as np
from utils.utils import mask_preprocess


# TODO: general
# TODO: pred_vol = np.zeros_like(vol[...,0]) --> X
def d2_model_inference(vol, batch_size, predictor):
    pred_vol = np.zeros_like(vol[...,0])
    for batch_start_index in range(0, vol.shape[0], batch_size):
        start, end = batch_start_index, min(vol.shape[0], batch_start_index+batch_size)
        img = vol[start:end]
        img_list = np.split(img, img.shape[0], axis=0)
        outputs = predictor(img_list) 

        for j, output in enumerate(outputs):
            pred = output["instances"]._fields['pred_masks'].cpu().detach().numpy() 
            pred = np.sum(pred, axis=0)
            pred = mask_preprocess(pred)
            img_idx = batch_start_index + j
            pred_vol[img_idx] = pred
    return pred_vol

            

# def pytorch_model_inference(vol, batch_size, predictor):
#     pred_vol = np.zeros_like(vol[...,0])
#     for batch_start_index in range(0, vol.shape[0], batch_size):
#         start, end = batch_start_index, min(vol.shape[0], batch_start_index+batch_size)
#         imgs = vol[start:end]
#         outputs = predictor(imgs) 
#         pred_vol[start:end] = outputs
#     return pred_vol




def pytorch_model_inference(predictor, dataloader):
    for idx, (vol, _) in enumerate(dataloader):
        vol = vol.to(torch.device('cuda:0'))
        vol = vol.to(torch.float)
        pred = predictor(vol[...,0])['out']
        pred = nn.Softmax(dim=1)(pred)
        pred = torch.argmax(pred, dim=1)
        if idx == 0:
            pred_vol = pred
        else:
            pred_vol = torch.cat([pred_vol, pred], 0)
    pred_vol = pred_vol.cpu().detach().numpy()
    return pred_vol


def inference_func(model_name):
    if model_name == '2D-Mask-RCNN':
        return d2_model_inference
    else:
        return pytorch_model_inference


def model_inference(model_name, vol, batch_size, predictor):
    inferencer = inference_func(model_name)
    return inferencer(vol, batch_size, predictor)

class Detectron2Inferencer():
    def __init__(self, cfg):
        self.model = BatchPredictor(cfg)

    def inference(self, inputs):
        predictions = []
        outputs = self.model(inputs)
        for j, output in enumerate(outputs):
            pred = output["instances"]._fields['pred_masks'].cpu().detach().numpy() 
            pred = np.sum(pred, axis=0)
            pred = mask_preprocess(pred)
            predictions.append(pred)
        return predictions


class BatchPredictor(DefaultPredictor):
    """Run d2 on a list of images."""
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, images):
        """Run d2 on a list of images.

        Args:
            images (list): BGR images of the expected shape: 720x1280
        """
        transform = []
        for origin_image in images:
            origin_image = origin_image[0]
            image = self.aug.get_transform(origin_image).apply_image(origin_image)
            transform.append({'image': torch.as_tensor(image.astype("float32").transpose(2, 0, 1)), 
                              'height': 512, 'width': 512})
        images = transform
        
        # images = [
        #     {'image': torch.as_tensor(image[0].astype("float32").transpose(2, 0, 1))}
        #     for image in images
        # ]
        with torch.no_grad():
            preds = self.model(images)
        return preds
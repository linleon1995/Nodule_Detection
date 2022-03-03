from detectron2.engine import DefaultPredictor
import torch
import numpy as np
from utils.utils import mask_preprocess


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
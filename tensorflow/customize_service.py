import logging
import threading

import numpy as np
import tensorflow as tf
from PIL import Image

from model_service.tfserving_model_service import TfServingBaseService
import os
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class flower_service(TfServingBaseService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.predict = None

        self.label_list = ['bee_balm', 'blackberry_lily', 'blanket_flower', 'bougainvillea', 'bromelia', 'foxglove']
        # self.label_list = ['foxglove', 'blackberry', 'blanket', 'bromelia', 'bee', 'bougainvillea']
        thread = threading.Thread(target=self.load_model)
        thread.start()

    def load_model(self):
        print('load model begin ======================================')
        self.model = tf.saved_model.load(self.model_path)

        signature_defs = self.model.signatures.keys()
        signature = []
        for signature_def in signature_defs:
            signature.append(signature_def)

        if len(signature) == 1:
            model_signature = signature[0]
        else:
            logging.warning("signatures more than one, use serving_default signature from %s", signature)
            model_signature = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        self.predict = self.model.signatures[model_signature]

    def _preprocess(self, data):
        images = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                image1 = Image.open(file_content)
                image1 = np.array(image1, dtype=np.float32)
                image1.resize((100, 100, 3))
                print(image1)
                images.append(image1)

        images = tf.convert_to_tensor(images, dtype=tf.dtypes.float32)
        preprocessed_data = images
        return preprocessed_data

    def _inference(self, data):
        return self.predict(data)

    def _postprocess(self, data):
        outputs = {}
        logits = data['dense_1'].numpy()[0].tolist()
        label_index = logits.index(max(logits))
        logits = ['%.4f' % logit for logit in logits]
        outputs['predicted_label'] = self.label_list[label_index]
        scores = dict(zip(self.label_list, logits))
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
        outputs['scores'] = scores
        return outputs

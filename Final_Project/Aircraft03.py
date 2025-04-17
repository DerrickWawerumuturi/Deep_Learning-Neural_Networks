## import the required libraries

import tensorflow as tf
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

## 2.1 Load BLIP Model
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

## Define the function for the model
class BlipModeL(processor, model):
    "class defines the blip model"
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def call(self, image_path, task):
        return tf.py_function(self.image_generator, [image_path, task], tf.string)

    def image_generator(self, image_path, task):
        image_path_str = image_path.numpy().decode('utf-8')

        image = Image.open(image_path_str).convert('RGB')

        if task.numpy().decode('utf-8') == 'caption':
            prompt = 'This is an image of '
        else:
            prompt = 'This is a detailed photo showing '

        inputs = self.processor(images=image, text=prompt, return_tensors='pt')
        output = self.model(**inputs)

        result = self.processor.decode(output[0], skip_special_tokens=True)
        return result

def text_generator(image_path, task):
    BlipMd = BlipModeL(processor, model)
    return BlipMd(image_path, task)

image_path = Image.open()
text_generator()
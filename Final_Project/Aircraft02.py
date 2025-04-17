#
# BLIP (Bootstrapping Language-Image Pretraining) is an advanced vision-and-language model designed to generate natural language descriptions for images.
# By leveraging both visual and textual information, BLIP can produce human-readable text that accurately reflects the content and context of an image.
# It is specifically trained to understand images and their relationships to summarizing text, making it ideal for tasks like image captioning, summarization, and visual question answering.
#
# In this project, learners will utilize the BLIP model to build a system capable of automatically generating captions and summary for images.
# The code will integrate the BLIP model within a custom Keras layer.
# This allows the user to input an image and specify a task, either "caption" or "summary", to receive a textual output that describes or summarizes the content of the image.

## Part 2: Image Captioning and Summarization using BLIP Pretrained Model

# Import required libraries

import tensorflow as tf
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 2.1: Load BLIP Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

# 2.2: Define Function for Image Captioning and Summarization
class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    def __init__(self, processor, model, **kwargs):
        """
        Initialize the custom Keras layer with the BLIP processor and model.

        Args:
            processor: The BLIP processor for preparing inputs for the model.
            model: The BLIP model for generating captions or summaries.
        """
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model

    def call(self, image_path, task):
        # Use tf.py_function to run the custom image processing and text generation
        return tf.py_function(self.process_image, [image_path, task], tf.string)

    def process_image(self, image_path, task):
        """
        Perform image loading, preprocessing, and text generation.

        Args:
            image_path: Path to the image file as a string.
            task: The type of task ("caption" or "summary").

        Returns:
            The generated caption or summary as a string.
        """
        try:
            # Decode the image path from the TensorFlow tensor to a Python string
            image_path_str = image_path.numpy().decode("utf-8")

            # Open the image using PIL and convert it to RGB format
            image = Image.open(image_path_str).convert('RGB')

            # Set the appropriate prompt based on the task
            
            if task.numpy().decode('utf-8') == 'caption':
                prompt = 'This is a picture of'
            else:
                prompt = 'This is a detailed photo showing'

            # prepare inputs for the BLIP model
            inputs = self.processor(images=image, text=prompt, return_tensors='pt')

            # Generate text output using the BLIP model
            output = self.model.generate(**inputs)

            # Decode the output into a readable string
            result = self.processor.decode(output[0], skip_special_tokens=True)
            return result

        except Exception as e:
            # Handle errors during image processing or text generation
            print(f'Errors: {e}')
            return 'Error processing image'

# 2.3: Create Custom Keras Layer for Image Captioning and Summarization
def generate_text(image_path, task):
    # Create a custom Keras layer for image captioning and summarization
    blip_layer = BlipCaptionSummaryLayer(processor, model)
    # Call the layer with the provided inputs
    return blip_layer(image_path, task)

## 2.4 Generating captions and summaries

# path to an example image
image_path = tf.constant("aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg")

# Generate a caption fot the image
caption = generate_text(image_path, tf.constant("caption"))

# Decode and print the generated caption
print("Caption:", caption.numpy().decode("utf-8"))

summary = generate_text(image_path, tf.constant("summary"))
# Decode and print the generated summary
print("Summary:", summary.numpy().decode("utf-8"))
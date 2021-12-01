import tflite_runtime.interpreter as tflite
import json
from keras_image_helper import create_preprocessor


preprocessor = create_preprocessor('xception', target_size=(150, 150))

interpreter = tflite.Interpreter(model_path='dogs_cats_10_0.687.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def predict(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    return pred

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return json.dumps(result, default=float)
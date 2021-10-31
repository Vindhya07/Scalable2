import tensorflow as tf
import tensorflow.keras as keras
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputModel', help='Name of Input Model', type=str)
    parser.add_argument('--outputLiteModel', help='Name of output Lite model', type=str)
    args = parser.parse_args()
    json_file = open(args.inputModel +'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(args.inputModel+'.h5')
    model.compile(loss='categorical_crossentropy',
                              optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                              metrics=['accuracy'])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
                    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
                    ]
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

                # Save the model.
    with open(args.outputLiteModel + '.tflite', 'wb') as f:
          f.write(tflite_model)

if __name__ == '__main__':
    main()

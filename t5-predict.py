from contextlib import contextmanager
import os
import tensorflow as tf
import t5

@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)

MODEL_SIZE = "3B" 
MODELS_DIR = "/nlrl/models-t5"
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)
DATA_DIR = "/nlrl/data"

# Set parallelism and batch size to fit on v2-8 TPU (if possible).
# Limit number of checkpoints to fit within 5GB (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = 1, 16, 1
batch_parallelism = 1

tf.io.gfile.makedirs(MODEL_DIR)

model = t5.models.MtfModel(
    model_dir=MODEL_DIR,
    tpu=None,
    mesh_shape = f'model:{model_parallelism},batch:{batch_parallelism}',
    mesh_devices = ["gpu:0"],
    batch_size=train_batch_size,
    sequence_length={"inputs": 128, "targets": 128},
    learning_rate_schedule=0.003,
    save_checkpoints_steps=5000,
    iterations_per_loop=100
)

predict_inputsraw_filename = 'prefixesraw.txt'
predict_inputsraw_path = os.path.join(DATA_DIR, predict_inputsraw_filename)


predict_inputs_filename = 'prefixes.txt'
predict_outputs_filename = 'predictions.txt'
predict_inputs_path = os.path.join(DATA_DIR, predict_inputs_filename)
predict_outputs_path = os.path.join(DATA_DIR, predict_outputs_filename)
# Manually apply preprocessing by prepending "triviaqa question:".

checkpoint = 1050000

prefixes = open(predict_inputsraw_path, 'r').readlines()
with tf.io.gfile.GFile(predict_inputs_path, "w") as f:
  for prefix in prefixes:
    f.write("initialstate&action: %s\n" % prefix.strip().lower())

# Ignore any logging so that we only see the model's answers to the questions.
with tf_verbosity_level('ERROR'):
  model.batch_size = 50
  model.predict(
      input_file=predict_inputs_path,
      output_file=predict_outputs_path,
      checkpoint_steps=checkpoint,
      # Select the most probable output token at each step.
      temperature=0
  )

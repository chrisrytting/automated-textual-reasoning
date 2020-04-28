import datetime
import functools
import json
import os
import re
import pprint
import random
import string
import sys
import tensorflow as tf

BASE_DIR = "."  # @param { type: "string" }
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models-t5")
ON_CLOUD = False

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import t5
import tensorflow as tf
import tensorflow_datasets as tfds
import time

# Improve logging.
from contextlib import contextmanager
import logging as py_logging


@contextmanager
def tf_verbosity_level(level):
    og_level = tf.logging.get_verbosity()
    tf.logging.set_verbosity(level)
    yield
    tf.logging.set_verbosity(og_level)


import gzip
import json

# =====================For converting JSONL to tsv===========================
## Public directory of Natural Questions data on GCS.

ATR_SPLIT_FNAMES = {
    "train": "atr-train.txt",
    "validation": "atr-validation.txt",
}
atr_counts_path = os.path.join(DATA_DIR, "atr-counts.json")
atr_tsv_path = {
    "train": os.path.join(DATA_DIR, "atr-train.tsv"),
    "validation": os.path.join(DATA_DIR, "atr-validation.tsv"),
}
#
#


def atr_txt_to_tsv(in_fname, out_fname):

    count = 0
    with open(in_fname, "r") as infile, open(out_fname, "w") as outfile:
        for line in infile:
            # Write this line as <question>\t<answer>
            is_a = re.search(".*Took[^/.]*", line).group(0)
            fs = line.replace(is_a, "")
            outfile.write("%s\t%s" % (is_a, fs))
            count += 1
            tf.logging.log_every_n(
                tf.logging.INFO, "Wrote %d examples to %s." % (count, out_fname), 1000
            )
        return count


if tf.io.gfile.exists(atr_counts_path):
    # Used cached data and counts.
    tf.logging.info("Loading ATR from cache.")
    num_atr_examples = json.load(tf.io.gfile.GFile(atr_counts_path))
else:
    # Create TSVs and get counts.
    tf.logging.info("Generating ATR TSVs.")
    num_atr_examples = {}

    for split, fname in ATR_SPLIT_FNAMES.items():
        num_atr_examples[split] = atr_txt_to_tsv(
            os.path.join(DATA_DIR, fname), atr_tsv_path[split]
        )
    json.dump(num_atr_examples, tf.io.gfile.GFile(atr_counts_path, "w"))


def atr_dataset_fn(split, shuffle_files=False):
    """
    Convert tsvs into tfds
    """
    # We only have one file for each split.
    del shuffle_files

    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(atr_tsv_path[split])
    # Split each "<question>\t<answer>" example into (question, answer) tuple.
    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=["", ""],
            field_delim="\t",
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # Map each tuple to a {"question": ... "answer": ...} dict.
    ds = ds.map(lambda *ex: dict(zip(["initialstate&action", "finalstate"], ex)))
    return ds


def atr_preprocessor(ds):
    """
    Convert tfds into a text-to-text format
    """

    def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, "'(.*)'", r"\1")
        return text

    def to_inputs_and_targets(ex):
        """Map {"initialstate&action": ..., "finalstate": ...}->{"inputs": ..., "targets": ...}."""
        return {
            "inputs": tf.strings.join(
                [
                    "initial state and action: ",
                    normalize_text(ex["initialstate&action"]),
                ]
            ),
            "targets": normalize_text(ex["finalstate"]),
        }

    return ds.map(
        to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


t5.data.TaskRegistry.add(
    "atr",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=atr_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[atr_preprocessor],
    # Use the same vocabulary that we used for pre-training.
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    num_input_examples=num_atr_examples,
)

atr_task = t5.data.TaskRegistry.get("atr")
ds = atr_task.get_dataset(
    split="validation", sequence_length={"inputs": 128, "targets": 32}
)
print("A few preprocessed validation examples...")
for ex in tfds.as_numpy(ds.take(5)):
    print(ex)


MODEL_SIZE = "3B"  # @param["small", "base", "large", "3B", "11B"]
# Public GCS path for T5 pre-trained model checkpoints
BASE_PRETRAINED_DIR = os.path.join(BASE_DIR, "base-t5")
PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)

if ON_CLOUD and MODEL_SIZE == "3B":
    tf.logging.warn(
        "The `3B` model is too large to use with the 5GB GCS free tier. "
        "Make sure you have at least 25GB on GCS before continuing."
    )
elif ON_CLOUD and MODEL_SIZE == "11B":
    raise ValueError(
        "The `11B` parameter is too large to fine-tune on the `v2-8` TPU "
        "provided by Colab. Please comment out this Error if you're running "
        "on a larger TPU."
    )

# Set parallelism and batch size to fit on v2-8 TPU (if possible).
# Limit number of checkpoints to fit within 5GB (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (2, 128, 8),
    "large": (8, 64, 4),
    "3B": (1, 1, 1),
    "11B": (8, 16, 1),
}[MODEL_SIZE]

batch_parallelism = 1

# ==================================================
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--stepsize', type = int)
parser.add_argument('--nsteps', type = int)
args = parser.parse_args()

# ==================================================

MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE, f"lr{args.lr}batch{args.batch_size}")
tf.io.gfile.makedirs(MODEL_DIR)
# The models from our paper are based on the Mesh Tensorflow Transformer.
model = t5.models.MtfModel(
    model_dir=MODEL_DIR,
    tpu=None,
    mesh_shape=f"model:{model_parallelism},batch:{batch_parallelism}",
    mesh_devices=["gpu:0"],
    batch_size=args.batch_size,
    sequence_length={"inputs": 250, "targets": 250},
    learning_rate_schedule=args.lr,
    save_checkpoints_steps=args.stepsize,
    iterations_per_loop=100,
)

import tensorboard as tb

tb.notebook.start("--logdir " + MODELS_DIR)

FINETUNE_STEPS = args.nsteps  # @param {type: "integer"}
import numpy as np


model.finetune(
    mixture_or_task_name="atr",
    pretrained_model_dir=PRETRAINED_DIR,
    finetune_steps=FINETUNE_STEPS,
)

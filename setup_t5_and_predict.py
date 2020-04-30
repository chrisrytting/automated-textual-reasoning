import t5
import os

def predict_from_input_file(
    model,
    experiment,
    input_file,
    output_file,
    checkpoints=-1,
    batch_size=16,
    temperature=0,
):

    model.batch_size = batch_size
    predict_inputs_path = os.path.join(
        "/nlrl/experiments/", experiment, f"testing/{input_file}"
    )
    predict_outputs_path = os.path.join(
        "/nlrl/experiments/", experiment, f"testing/{output_file}"
    )

    if isinstance(checkpoints, list):
        checkpoints = [int(ckpt) for ckpt in checkpoints]

    model.predict(
        input_file=predict_inputs_path,
        output_file=predict_outputs_path,
        checkpoint_steps=checkpoints,
        # Select the most probable output token at each step.
        temperature=temperature,
    )

def setup_t5_and_predict(
    model_dir="/nlrl/models-t5/3B",
    input_file="prefixes.txt",
    output_file="predictions.txt",
    checkpoints=[-1],
    model_parallelism=1,
    batch_parallelism=1,
    gpu_ids=[0],
    experiment="experiment1",
    train_batch_size=32,
    temperature=0.0,
):
    """
    This file sets up a t5 model found in model_dir and predicts from 
    input_file. The resulting predictions are found in output_file, where the
    checkpoint used to generate them is appended to the filename.
    """

    for ix, ckpt in enumerate(checkpoints):

        model = t5.models.MtfModel(
            model_dir=model_dir,
            tpu=None,
            mesh_shape=f"model:{model_parallelism},batch:{batch_parallelism}",
            mesh_devices=[f'gpu:{ix}' for gpu in gpu_ids],
            batch_size=train_batch_size,
            sequence_length={"inputs": 250, "targets": 250},
            iterations_per_loop=100,
        )

        predict_from_input_file(
            model,
            experiment,
            input_file,
            output_file,
            checkpoints=ckpt,
            temperature=temperature,
        )

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=int)
    parser.add_argument('--gpu_id', type=int)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--model_dir', type=str)
    args = parser.parse_args()


    setup_t5_and_predict(model_dir=args.model_dir,checkpoints = [args.ckpt], gpu_ids = [args.gpu_id], experiment=args.experiment)

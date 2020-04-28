
def setup_t5_and_predict(
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

    for ix, ckpt in enumerate(checkpoints):

        model = t5.models.MtfModel(
            model_dir="/nlrl/models-t5/3B",
            tpu=None,
            mesh_shape=f"model:{model_parallelism},batch:{batch_parallelism}",
            mesh_devices=[f'gpu:{ix}' for gpu in gpu_ids],
            batch_size=train_batch_size,
            sequence_length={"inputs": 250, "targets": 250},
            iterations_per_loop=100,
        )

        gt.predict_from_input_file(
            model,
            experiment,
            input_file,
            output_file,
            checkpoints=ckpt,
            temperature=temperature,
        )

if __name__=="__main__":
    import argparse
    import generate_templates as gt
    import t5

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=int)
    parser.add_argument('--gpu_id', type=int)
    parser.add_argument('--experiment', type=str)
    args = parser.parse_args()


    setup_t5_and_predict(checkpoints = [args.ckpt], gpu_ids = [args.gpu_id], experiment=args.experiment)

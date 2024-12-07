from __future__ import absolute_import
import random
import torch.optim as optim

import mlflow
from tools import *
from models import create_net


def load_params_from_mlflow(run_id):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.params


def train_model(args, run_id=None):
    train_loader, val_loader, test_loader = CIFAR_data_loaders(args)
    model = create_net(args)
    # print(model)
    # Select optimizer based on args.optim
    if args.optim.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                              momentum=args.beta1, weight_decay=args.weight_decay)
    elif args.optim.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(
            args.beta1, args.beta2), weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optim}")

    model_trainer = CIFAR_Trainer(
        args, model, optimizer, train_loader, val_loader, test_loader)

    if args.resume:
        checkpoint_path = find_last_checkpoint(args.ckpt_path)
        model, optimizer, best_acc, best_acc_5, start_epoch = load_checkpoint(
            args, model, checkpoint_path, optimizer)
    else:
        start_epoch = 0
        best_acc = 0
        best_acc_5 = 0

    if len(args.gpu_ids) > 0:
        model.to(args.gpu_ids[0])
        model = torch.nn.DataParallel(model, args.gpu_ids)  # multi-GPUs

    args.log_file.write("Network - " + args.arch + "\n")
    args.log_file.write("Attention Module - " + args.attention_type + "\n")
    args.log_file.write(str(model))
    args.log_file.write(
        "--------------------------------------------------" + "\n")

    is_best_top1 = False
    is_best_top5 = False

    best_acc = 0
    best_acc_5 = 0

    for epoch in range(start_epoch, args.num_epoch):

        model_trainer.adjust_learning_rate(epoch)

        model_trainer.train_epoch(epoch)
        val_acc, val_acc_5 = model_trainer.validate(epoch)
        if epoch >= args.start_validation_epoch:

            # Determine if this is the best model
            is_best_top1 = val_acc > best_acc
            is_best_top5 = val_acc_5 > best_acc_5
            best_acc = max(val_acc, best_acc)
            best_acc_5 = max(val_acc_5, best_acc_5)

        save_checkpoint(args, model_trainer.model, model_trainer.optimizer, is_best_top1,
                        is_best_top5, best_acc, best_acc_5, epoch, run_id, save_path=args.ckpt_path)

        args.log_file.write(
            "--------------------------------------------------" + "\n")

    args.log_file.write(
        "Best top-1 accuracy on validation set: %4.2f\n" % best_acc)
    args.log_file.write(
        "Best top-5 accuracy on validation set: %4.2f\n" % best_acc_5)

    if args.mlflow_logging:
        mlflow.log_metric('Validation/best top-1 accuracy', best_acc)
        mlflow.log_metric('Validation/best top-5 accuracy', best_acc_5)

    if args.validation:

        best_top1_path = os.path.join(
            args.ckpt_path, 'model_best_top1.pth.tar')
        model, _, best_acc, best_acc_5, start_epoch = load_checkpoint(
            args, model, best_top1_path)
        test_top1_acc = model_trainer.test(model, top_k=1)

        best_top5_path = os.path.join(
            args.ckpt_path, 'model_best_top5.pth.tar')
        model, _, best_acc, best_acc_5, start_epoch = load_checkpoint(
            args, model, best_top5_path)
        test_top5_acc = model_trainer.test(model, top_k=5)

        args.log_file.write(
            "Test accuracy with best top-1 model: %4.2f\n" % test_top1_acc)
        args.log_file.write(
            "Test accuracy with best top-5 model: %4.2f\n" % test_top5_acc)

        if args.mlflow_logging:
            mlflow.log_metric('Test/best top-1 accuracy', test_top1_acc)
            mlflow.log_metric('Test/best top-5 accuracy', test_top5_acc)

        avg_fps = model_trainer.test_inference_speed()

        print(f'Average Inference FPS: {avg_fps:.2f}')

    if args.mlflow_logging:
        model_string = args.attention_type+'-'+args.arch+'-'+args.dataset
        mlflow.pytorch.log_model(
            model, model_string, signature=model_trainer.signature)

    if args.tensorboard_logging:
        absolute_tensorboard_path = os.path.abspath(args.tensorboard_path)

        print(
            "\nLaunch TensorBoard with:\n\n"
            "tensorboard --logdir={}".format(absolute_tensorboard_path)
        )
        model_trainer.writer.close()
        if args.mlflow_logging:
            print("Uploading TensorBoard events as a mlrun artifact...")
            mlflow.log_artifacts(args.tensorboard_path, artifact_path="events")
    print("===================================Job Done!===========================================")


if __name__ == "__main__":

    args = parse_config()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_string = args.dataset

    train_string += "-" + args.arch
    # train_string += "-" + args.block_type
    if args.attention_type.lower() != "none":
        train_string += "-" + args.attention_type

    # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_string += "-" + timestamp

    if args.mlflow_logging:
        args.mlflow_log_dir = os.path.abspath(args.mlruns_path)
        mlflow.set_tracking_uri(f"file://{args.mlflow_log_dir}")
        mlflow.set_experiment(args.experiment)

        if args.resume:
            checkpoint_path = find_last_checkpoint(args.ckpt_path)
            run_id = load_run_id_from_checkpoint(checkpoint_path)
            # Load parameters from MLflow and update args
            params = load_params_from_mlflow(run_id)
            check_args(args, params)

            with mlflow.start_run(run_id=run_id, run_name=train_string) as run:
                train_model(args, run_id)
        else:
            run_id = None

            with mlflow.start_run(run_name=train_string) as run:
                run_id = run.info.run_id
                # Set multiple tags. Each tag must have a key and a value.
                mlflow.set_tag("architecture", args.arch)
                # mlflow.set_tag("block_type", args.block_type)
                mlflow.set_tag("dataset", args.dataset)
                mlflow.set_tag("attention_type", args.attention_type)

                for key, val in vars(args).items():
                    mlflow.log_param(key, val)
                    print("{:16} {}".format(key, val))

                train_model(args, run_id)

    else:
        train_model(args)

    args.log_file.close()

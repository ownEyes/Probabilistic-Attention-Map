import datetime
import os
import yaml
import argparse
import sys


def parse_gpus(gpu_ids):
    gpus = gpu_ids.split(',')
    gpu_ids = []
    for g in gpus:
        g_int = int(g)
        if g_int >= 0:
            gpu_ids.append(g_int)
    if not gpu_ids:
        return None
    return gpu_ids


def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The configuration file {config_path} was not found.")
        exit(1)
    except yaml.YAMLError as error:
        print(f"Error parsing configuration file: {error}")
        exit(1)


def validate_config(config):
    required_keys = ["batch_size", "dataset", "base_lr"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")


def parse_config():
    parser = argparse.ArgumentParser(
        description="Load YAML config for training")
    parser.add_argument("config", type=str,
                        help="Path to the configuration YAML file")

    args = parser.parse_args()

    # Load the YAML file using the config path
    if os.path.exists(args.config):
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            # Update the args namespace with values from the YAML file
            args.__dict__.update(config)
    else:
        print(f"Configuration file not found at {args.config}")
        sys.exit()

    args.gpu_ids = parse_gpus(args.gpu_ids)

    train_string = args.dataset

    train_string += "-" + args.arch
    # train_string += "-" + args.block_type
    if args.attention_type.lower() != "none":
        train_string += "-" + args.attention_type
        train_string += "-param" + str(args.attention_param)
        train_string += "-paramTwo" + str(args.attention_param2)

    if args.other_mark.lower() != "none":
        train_string += "-mark_" + str(args.other_mark)

    # train_string += "-nfilters" + str(args.num_base_filters)
    # train_string += "-expansion" + str(args.expansion)
    if args.validation:
        train_string += "-val_size" + str(args.val_size)
    train_string += "-baselr" + str(args.base_lr)
    train_string += "-rseed" + str(args.seed)

    args.ckpt_path += train_string
    args.log_path += train_string

    if args.tensorboard_logging:
        # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # train_string += "-" + timestamp  # Append the timestamp
        args.tensorboard_path += train_string

    if not os.path.isdir(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)
    if args.tensorboard_logging and not os.path.isdir(args.tensorboard_path):
        os.makedirs(args.tensorboard_path)
    if args.mlflow_logging:
        if not os.path.isdir(args.mlflow_path):
            os.makedirs(args.mlflow_path)
        # Append 'mlruns' to the path
        args.mlruns_path = os.path.join(args.mlflow_path, 'mlruns')
        # Ensure the 'mlruns' subfolder exists
        if not os.path.isdir(args.mlruns_path):
            os.makedirs(args.mlruns_path)

    # write to file
    args.log_file = open(os.path.join(
        args.log_path, "log_file.txt"), mode="a")

    return args


class ConfigMismatchException(Exception):
    pass


def check_args(args, params):
    for key, value in params.items():
        if key == 'resume':
            continue
        if hasattr(args, key):
            current_value = getattr(args, key)
            if str(current_value) != value:
                raise ConfigMismatchException(
                    f"Argument '{key}' has a different value. Current: {
                        current_value}, Stored: {value}"
                )
        else:
            raise ConfigMismatchException(
                f"Argument '{key}' not found in args."
            )

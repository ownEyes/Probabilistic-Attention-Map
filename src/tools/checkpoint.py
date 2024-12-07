import os
import torch
import shutil


def save_checkpoint(args, model, optimizer, is_best_top1, is_best_top5, best_acc, best_acc_5, epoch, run_id, save_path='./'):
    print("=> saving checkpoint '{}'".format(epoch))
    # Extract state_dict from the model
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.cpu().state_dict()
    else:
        state_dict = model.cpu().state_dict()

    state = {
        "epoch": epoch + 1,
        "arch": args.arch,
        "state_dict": state_dict,
        "best_acc": best_acc,
        "best_acc_5": best_acc_5,
        "optimizer": optimizer.state_dict(),
        "mlflow_run_id": run_id,
    }

    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    if (epoch % 10 == 0):
        torch.save(state, os.path.join(
            save_path, 'checkpoint_%03d.pth.tar' % epoch))

    if is_best_top1:
        shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'),
                        os.path.join(save_path, 'model_best_top1.pth.tar'))

    if is_best_top5:
        shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'),
                        os.path.join(save_path, 'model_best_top5.pth.tar'))

    # Ensure the model is back on the device
    model.to(args.device)


def load_checkpoint(args, model, checkpoint_path, optimizer=None, verbose=True):

    # checkpoint_path = find_last_checkpoint(args.ckpt_path)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0)
    best_acc_5 = checkpoint.get('best_acc_5', 0)

    # Load state_dict with potential key adjustments
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items(
    ) if k.startswith('module.') or k in model.state_dict()}
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)

    if verbose:
        print("=> loading checkpoint '{}' (epoch {})"
              .format(checkpoint_path, start_epoch))

    return model, optimizer, best_acc, best_acc_5, start_epoch


def find_last_checkpoint(save_path):
    files = [f for f in os.listdir(
        save_path) if 'checkpoint' in f and f.endswith('.pth.tar')]
    if files:
        files.sort(key=lambda f: os.path.getmtime(
            os.path.join(save_path, f)), reverse=True)
        return os.path.join(save_path, files[0])
    return None


def load_run_id_from_checkpoint(checkpoint_path, verbose=True):
    """
    Load the MLflow run ID from a checkpoint file.

    Parameters:
    checkpoint_path (str): Path to the checkpoint file.
    verbose (bool): If True, print the run ID being loaded.

    Returns:
    str: The MLflow run ID if present in the checkpoint, otherwise None.
    """

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Extract the run_id from the checkpoint
    run_id = checkpoint.get('mlflow_run_id', None)

    if verbose:
        if run_id:
            print(f"=> Loaded MLflow run ID '{
                  run_id}' from checkpoint '{checkpoint_path}'")
        else:
            print(f"=> No MLflow run ID found in checkpoint '{
                  checkpoint_path}'")

    return run_id

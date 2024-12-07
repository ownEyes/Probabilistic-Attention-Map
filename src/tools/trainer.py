import torch
import time
import types
from ptflops import get_model_complexity_info
from tensorboardX import SummaryWriter
# from thop import profile
from torch.nn import functional as F
import mlflow
from mlflow.models.signature import infer_signature

from .util import AverageMeter, ProgressMeter, accuracy


class CIFAR_Trainer:
    def __init__(self, config, model, optimizer, train_loader, val_loader, test_loader):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.writer = None
        if self.config.tensorboard_logging:
            self.writer = SummaryWriter(log_dir=self.config.tensorboard_path)
            # self.model.log_weights = types.MethodType(log_weights, model)

        # Move the model to the appropriate device
        self.model.to(self.config.device)

        # Get a single batch from the training loader
        try:
            data, _ = next(iter(self.train_loader))
            # Use a single data example
            dummy_input = data[0:1].to(self.config.device)

            if self.writer:
                self.writer.add_graph(self.model, (dummy_input,))

            self.model.eval()
            with torch.no_grad():
                sample_output = self.model(dummy_input)

            # Calculate FLOPs
            # flops, params = profile(self.model, inputs=(dummy_input, ))
            flops, params = get_model_complexity_info(self.model, (3, 32, 32), as_strings=False, backend='pytorch',
                                                      print_per_layer_stat=True, verbose=True)

            if self.writer:
                self.writer.add_scalar('FLOPs', flops)
                self.writer.add_scalar('Params', params)

            if self.config.mlflow_logging:
                mlflow.log_metric("FLOPs", flops)
                mlflow.log_metric("Params", params)

                dummy_input = dummy_input.to("cpu")
                sample_output = sample_output.to("cpu")
                self.signature = infer_signature(
                    dummy_input.numpy(), sample_output.numpy())
        except Exception as e:
            print(f"Failed to log model architecture: {e}")

    def adjust_learning_rate(self, epoch):
        """Adjust the learning rate based on epoch and configuration."""
        if epoch <= self.config.lr_decay_start:
            lr = 0.01 if self.config.warmup and epoch < 3 else self.config.base_lr
        elif epoch <= self.config.lr_decay_end:
            lr = self.config.base_lr * 0.1
        else:
            lr = self.config.base_lr * 0.01

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def log_weights(self, step):
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f"weights/{name}", param.data, step)
            if param.grad is not None:
                self.writer.add_histogram(
                    f"grads/{name}", param.grad.data, step)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        learning_rate = self.optimizer.param_groups[0]["lr"]

        batch_time = AverageMeter('Batch Time', ':6.3f')
        data_time = AverageMeter('Data Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Accuracy', ':4.2f')

        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, top1],
            prefix="Epoch (Train LR {:6.4f}): [{}] ".format(learning_rate, epoch))

        self.model.train()

        tic = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):

            data, target = data.to(self.config.device, non_blocking=True), target.to(
                self.config.device, non_blocking=True)

            data_time.update(time.time() - tic)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            acc = accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc[0].item(), data.size(0))

            batch_time.update(time.time() - tic)
            tic = time.time()

            if self.config.tensorboard_logging:
                # Log metrics to TensorBoard
                self.writer.add_scalar(
                    'Training/Loss', losses.val, epoch * len(self.train_loader) + batch_idx)
                self.writer.add_scalar(
                    'Training/Accuracy', top1.val, epoch * len(self.train_loader) + batch_idx)

                self.writer.add_scalar(
                    'Training/Learning Rate', self.optimizer.param_groups[0]["lr"], epoch * len(self.train_loader) + batch_idx)

            if self.config.mlflow_logging:
                mlflow.log_metric('Training/Loss', losses.val,
                                  epoch * len(self.train_loader) + batch_idx)
                mlflow.log_metric('Training/Accuracy', top1.val,
                                  epoch * len(self.train_loader) + batch_idx)
                mlflow.log_metric('Training/Learning Rate',
                                  self.optimizer.param_groups[0]["lr"], epoch * len(self.train_loader) + batch_idx)

            if (batch_idx + 1) % self.config.disp_iter == 0 or (batch_idx + 1) == len(self.train_loader):
                epoch_msg = progress.get_message(batch_idx + 1)
                print(epoch_msg)
                if self.config.tensorboard_logging:
                    self.log_weights(
                        epoch * len(self.train_loader) + batch_idx)
                self.config.log_file.write(epoch_msg + "\n")

    def validate(self, epoch):
        """Validate the model performance."""
        batch_time = AverageMeter('Batch Time', ':6.3f')
        data_time = AverageMeter('Data Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':4.2f')
        top5 = AverageMeter('Acc@5', ':4.2f')

        progress = ProgressMeter(
            len(self.val_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch (Valid LR {:6.4f}): [{}] ".format(0, epoch))

        self.model.eval()

        with torch.no_grad():
            tic = time.time()
            for batch_idx, (data, target) in enumerate(self.val_loader):

                data, target = data.to(self.config.device, non_blocking=True), target.to(
                    self.config.device, non_blocking=True)

                data_time.update(time.time() - tic)

                output = self.model(data)

                if batch_idx == 0:
                    probabilities = F.softmax(output, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    if self.config.tensorboard_logging:
                        # Log only the first batch of every epoch
                        # Iterate through each image in the batch
                        for i in range(data.size(0)):
                            image = data[i]
                            label = target[i]
                            pred = predictions[i]

                            # Re-normalize and clamp the image for visualization
                            mean = torch.tensor(
                                [0.4914, 0.4822, 0.4465], device=self.config.device).view(1, 3, 1, 1)
                            std = torch.tensor(
                                [0.247, 0.243, 0.261], device=self.config.device).view(1, 3, 1, 1)
                            image = image * std + mean
                            image = torch.clamp(image, 0, 1)

                            # Remove the batch dimension since it's a single image (squeeze only if batch size is 1)
                            # This changes the shape from (1, 3, 32, 32) to (3, 32, 32)
                            image = image.squeeze(0)

                            # Log image with prediction and label
                            self.writer.add_image(
                                f'Validation/Image_{i}_Label_{label.item()}_Pred_{pred.item()}', image, epoch)

                loss = F.cross_entropy(output, target)

                acc = accuracy(output, target)
                acc_5 = accuracy(output, target, (5,))

                losses.update(loss.item(), data.size(0))

                top1.update(acc[0].item(), data.size(0))
                top5.update(acc_5[0].item(), data.size(0))

                batch_time.update(time.time() - tic)
                tic = time.time()

                if self.config.tensorboard_logging:
                    # Log validation metrics to TensorBoard
                    self.writer.add_scalar(
                        'Validation/Loss', losses.val, epoch * len(self.val_loader) + batch_idx)
                    self.writer.add_scalar(
                        'Validation/Acc-1', top1.val, epoch * len(self.val_loader) + batch_idx)

                    self.writer.add_scalar(
                        'Validation/Acc-5', top5.val, epoch * len(self.val_loader) + batch_idx)

                if self.config.mlflow_logging:
                    mlflow.log_metric('Validation/Loss', losses.val,
                                      epoch * len(self.train_loader) + batch_idx)
                    mlflow.log_metric('Validation/Acc-1', top1.val,
                                      epoch * len(self.train_loader) + batch_idx)
                    mlflow.log_metric('Validation/Acc-5', top5.val,
                                      epoch * len(self.train_loader) + batch_idx)

            if (batch_idx + 1) % self.config.disp_iter == 0 or (batch_idx + 1) == len(self.val_loader):
                epoch_msg = progress.get_message(batch_idx + 1)
                print(epoch_msg)
                if self.config.tensorboard_logging:
                    self.log_weights(
                        epoch * len(self.train_loader) + batch_idx)
                self.config.log_file.write(epoch_msg + "\n")

            print(
                '-------- Mean Acc@1 {top1.avg:.3f} --------'.format(top1=top1))
            print(
                '-------- Mean Acc@5 {top5.avg:.3f} --------'.format(top5=top5))

        return top1.avg, top5.avg

    def test(self, model, top_k=1):
        """Test the model performance on the test set."""
        batch_time = AverageMeter('Batch Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        topk = AverageMeter(f'Acc@{top_k}', ':4.2f')

        progress = ProgressMeter(
            len(self.test_loader),
            [batch_time, losses, topk],
            prefix="Test: "
        )

        model.eval()

        with torch.no_grad():
            tic = time.time()
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.config.device, non_blocking=True), target.to(
                    self.config.device, non_blocking=True)

                output = model(data)
                loss = F.cross_entropy(output, target)

                acc_k = accuracy(output, target, (top_k,))

                losses.update(loss.item(), data.size(0))
                topk.update(acc_k[0].item(), data.size(0))

                batch_time.update(time.time() - tic)
                tic = time.time()

                if self.config.tensorboard_logging:
                    # Log test metrics to TensorBoard
                    self.writer.add_scalar('Test/Loss', losses.val, batch_idx)
                    self.writer.add_scalar(
                        f'Test/Acc-{top_k}', topk.val, batch_idx)

                if self.config.mlflow_logging:
                    mlflow.log_metric('Test/Loss', losses.val, batch_idx)
                    mlflow.log_metric(f'Test/Acc-{top_k}', topk.val, batch_idx)

                if (batch_idx + 1) % self.config.disp_iter == 0 or (batch_idx + 1) == len(self.test_loader):
                    print(progress.get_message(batch_idx + 1))

            print(
                f'-------- Mean Acc@{top_k} {topk.avg:.3f} --------'.format(topk=topk))

        return topk.avg

    def test_inference_speed(self, num_batches=10):
        """Test the inference speed by forwarding some images from the dataset and report the average FPS."""
        self.model.eval()
        start_time = time.time()

        total_images = 0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.test_loader):
                if batch_idx >= num_batches:
                    break
                data = data.to(self.config.device, non_blocking=True)
                batch_size = data.size(0)
                total_images += batch_size

                output = self.model(data)

        elapsed_time = time.time() - start_time
        avg_fps = total_images / elapsed_time

        # Log FPS to TensorBoard and MLflow
        if self.config.tensorboard_logging:
            self.writer.add_scalar('Inference/FPS', avg_fps)
        if self.config.mlflow_logging:
            mlflow.log_metric('Inference/FPS', avg_fps)

        return avg_fps

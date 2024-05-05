import torch
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
from datasets.suas import mapping

import wandb

# define step counter globally
step = 0

def topK(out, labels, TOPK_N=5):
    _, topk_out = torch.topk(out, TOPK_N)
    topk_conf = (topk_out == labels.unsqueeze(1)).any(1).float().mean().item()
    local_pred = np.vectorize(lambda x: mapping[x])(topk_out)
    local_label = np.vectorize(lambda x: mapping[x])(labels.cpu().unsqueeze(1))

    return topk_conf, np.hstack((local_pred, local_label))


def train(model, optim, scheduler, criterion, train_set, device, epochs=2, grad_clip_value=1.0, print_interval=100,
          model_save_path="./ckgcnn.pt", save_interval=100, test_fn=None, global_stepcount=True):
    """

    :param model:
    :param optim:
    :param criterion:
    :param train_set:
    :param device:
    :param epochs:
    :param grad_clip_value:
    :param print_interval:
    :param model_save_path:
    :param save_interval:
    :param test_fn:
    :param global_stepcount:
    """

    writer = SummaryWriter()
    total_samples = 0
    if global_stepcount:
        global step
    else:
        step = 0

    best_acc = 0.

    for epoch in range(epochs):
        model.to(device)
        model.train()

        # Accumulate accuracy and loss
        running_loss = 0
        running_corrects = 0

        for iteration, (samples, labels) in enumerate(train_set):
            optim.zero_grad()

            samples = samples.to(device)
            labels = labels.to(device)

            torch.cuda.memory._record_memory_history(max_entries=100000)
            # forward pass
            out = model(samples)
            loss = criterion(out, labels)
            torch.cuda.memory._dump_snapshot("out.pkl")
            torch.cuda.memory._record_memory_history(enabled=None)

            # backward pass, gradient clipping and weight update
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), grad_clip_value
            # )
            optim.step()

            # keep track of running loss and correctly classified samples
            running_loss += (loss.item() * labels.size(0))
            corrects = (torch.max(out, 1)[1] == labels).sum().item()


            

            running_corrects += corrects
            step += 1
            total_samples += labels.size(0)

            # print running loss every n steps
            if not iteration % print_interval:
                writer.add_scalar("Loss/train", loss, step)
                writer.add_scalar("Accuracy/train", corrects/labels.size(0), step)

                print("TOPK")
                topk_conf, fused_aryy = topK(out.cpu(), labels.cpu())
                print(fused_aryy)
                print(topk_conf)
                print("LOGITS")
                local_out = np.vectorize(lambda x: mapping[x])(torch.argmax(out.cpu(), 1).numpy())
                local_label = np.vectorize(lambda x: mapping[x])(labels.cpu().numpy())
                print(local_out)
                print(local_label)
                TOPK_N = 5
                writer.add_scalar(f"top_{TOPK_N}_accuracy/train", topk_conf, step)
                writer.flush()


                if wandb.run:
                    wandb.log({"epoch": epoch, "loss": loss.item(), "batch_accuracy": corrects/labels.size(0)}, step=step)
                print(f"epoch {epoch} - iteration {iteration} - batch loss {loss.item():.2f} - batch accuracy {corrects / labels.size(0):.2f}")
                print()

            # save the model on interval
            if not iteration % save_interval:
                if model_save_path:
                    torch.save(model, model_save_path)

        # save the model after each epoch
        if model_save_path:
            torch.save(model, model_save_path)

        if test_fn:
            val_acc = test_fn()

            if val_acc > best_acc:
                best_acc = val_acc

                if wandb.run:
                    wandb.log({"best_accuracy": best_acc})

            # step learning rate
            if scheduler:
                scheduler.step()

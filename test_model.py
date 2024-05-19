import torch
import wandb
import torch
from tqdm import tqdm


def test(model, test_set, device, step, loss=None, limit=None, writer=None, test_name = "test"):
    """
    Evaluate the classification accuracy of a given model on a given test set.

    :param model:
    :param test_set:
    :return:
    """
    correct = 0
    total = 0
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        model.eval()
        for data in tqdm(test_set):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            total += labels.size(0)
            correct += (torch.max(out, 1)[1] == labels).sum().item()
            total_batches += 1

            if loss:
                total_loss += loss(out, labels).item()

            if limit:
                if total > limit:
                    break

    if writer:
        writer.add_scalar(f"Loss/{test_name}", total_loss, step)
        writer.add_scalar(f"Accuracy/{test_name}", 100* correct / total, step)
        writer.flush()
    print(f"{test_name} set accuracy on {total} samples: {(100 * correct / total):.2f}%")

    if wandb.run:
        wandb.log({"test_accuracy": correct/total, "test_loss": total_loss/total_batches})
    return correct / total


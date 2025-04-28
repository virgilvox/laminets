import torch
from train_laminet_micro import LaminetMicro, LaminetDataset
from torch.utils.data import DataLoader

def evaluate():
    dataset = LaminetDataset('laminet_synthetic_dataset.json')
    loader = DataLoader(dataset, batch_size=8)

    model = LaminetMicro().cuda()
    model.load_state_dict(torch.load('laminet_micro_checkpoint.pth'))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs, _ = model(inputs)
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate()

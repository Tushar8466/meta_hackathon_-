import torch

def choose_action(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        output = model(image)
        action = torch.argmax(output, dim=1).item()
    return action
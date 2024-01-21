import torch.nn.functional as F

def one_hot_encode(labels, num_classes=6):
    # labels: (batch_size, seq_length)
    # One-hot encode and then unsqueeze to add channel dimension
    return F.one_hot(labels, num_classes=num_classes).float()

def repeat_labels(labels, repeat_factor=30):
    # Repeat each label repeat_factor times along the sequence length dimension
    return labels.repeat_interleave(repeat_factor, dim=1)
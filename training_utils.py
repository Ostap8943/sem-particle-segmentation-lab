import cv2
import numpy as np
import torch
import torch.nn as nn


def create_manual_mask(image_shape, polygons):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        if len(poly) >= 3:
            cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
    return mask


def augment_pairs(image, mask, use_augmentation):
    pairs = [(image, mask)]
    if use_augmentation:
        pairs.extend(
            [
                (np.fliplr(image), np.fliplr(mask)),
                (np.flipud(image), np.flipud(mask)),
                (np.rot90(image, 1), np.rot90(mask, 1)),
                (np.rot90(image, 2), np.rot90(mask, 2)),
                (np.rot90(image, 3), np.rot90(mask, 3)),
                rotate_pair(image, mask, -20),
                rotate_pair(image, mask, 20),
                (np.clip(image * 1.12 + 0.03, 0.0, 1.0), mask),
                (np.clip(image * 0.88 - 0.02, 0.0, 1.0), mask),
                (np.clip(image + np.random.normal(0.0, 0.025, image.shape), 0.0, 1.0), mask),
            ]
        )
    return pairs


def rotate_pair(image, mask, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_image, rotated_mask


def build_training_batch(image, mask, use_augmentation):
    pairs = augment_pairs(image, mask, use_augmentation)

    images = np.stack([img.astype(np.float32) for img, _ in pairs])
    masks = np.stack([msk.astype(np.float32) for _, msk in pairs])
    img_tensor = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    mask_tensor = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)
    return img_tensor, mask_tensor


def crop_particle_patch(gray, mask, polygon, patch_size=256, padding=2.6):
    pts = np.array(polygon, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    side = int(max(36, max(w, h) * padding))
    cx = x + w // 2
    cy = y + h // 2
    half = side // 2

    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + side, y1 + side

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - gray.shape[1])
    pad_bottom = max(0, y2 - gray.shape[0])

    if pad_left or pad_top or pad_right or pad_bottom:
        gray = cv2.copyMakeBorder(gray, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    img_patch = gray[y1:y2, x1:x2]
    mask_patch = mask[y1:y2, x1:x2]
    img_patch = cv2.resize(img_patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    mask_patch = cv2.resize(mask_patch, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    return img_patch.astype(np.float32) / 255.0, mask_patch.astype(np.float32) / 255.0


def build_particle_patch_batch(cv_image, manual_mask, polygons, use_augmentation, max_samples=72):
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    pairs = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        patch_img, patch_mask = crop_particle_patch(gray, manual_mask, polygon)
        if np.count_nonzero(patch_mask) == 0:
            continue
        pairs.extend(augment_pairs(patch_img, patch_mask, use_augmentation))

    if not pairs:
        input_img = cv2.resize(gray, (256, 256)).astype(np.float32) / 255.0
        input_mask = cv2.resize(manual_mask, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        pairs = augment_pairs(input_img, input_mask, use_augmentation)

    if len(pairs) > max_samples:
        step = len(pairs) / max_samples
        pairs = [pairs[int(i * step)] for i in range(max_samples)]

    images = np.stack([img.astype(np.float32) for img, _ in pairs])
    masks = np.stack([msk.astype(np.float32) for _, msk in pairs])
    img_tensor = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    mask_tensor = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)
    return img_tensor, mask_tensor


def train_unet(model, cv_image, manual_mask, polygons, epochs, learning_rate, use_augmentation, status_callback=None):
    img_tensor, mask_tensor = build_particle_patch_batch(cv_image, manual_mask, polygons, use_augmentation)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    positive_pixels = torch.clamp(mask_tensor.sum(), min=1.0)
    negative_pixels = torch.clamp(mask_tensor.numel() - mask_tensor.sum(), min=1.0)
    pos_weight = torch.clamp(negative_pixels / positive_pixels, min=1.0, max=8.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size = min(16, img_tensor.shape[0])
    last_loss = 0.0

    for epoch in range(int(epochs)):
        order = torch.randperm(img_tensor.shape[0])
        epoch_losses = []
        for start in range(0, img_tensor.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            batch_img = img_tensor[batch_idx]
            batch_mask = mask_tensor[batch_idx]
            optimizer.zero_grad()
            outputs = model(batch_img)
            bce_loss = criterion(outputs, batch_mask)
            dice_loss = soft_dice_loss(outputs, batch_mask)
            loss = bce_loss + dice_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        last_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if status_callback and (epoch % 20 == 0 or epoch == int(epochs) - 1):
            status_callback(epoch, int(epochs), last_loss, img_tensor.shape[0])

    return last_loss, img_tensor.shape[0]


def soft_dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

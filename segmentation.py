import cv2
import numpy as np
import torch


U_NET_METHOD = "U-Net neural network"


def normalized_odd(value):
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def choose_foreground(mask, gray=None):
    if gray is not None:
        selected = gray[mask > 0]
        rejected = gray[mask == 0]
        if selected.size and rejected.size and selected.mean() < rejected.mean():
            mask = cv2.bitwise_not(mask)
        return mask

    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if np.mean(border) > 127:
        mask = cv2.bitwise_not(mask)
    return mask


def extract_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def segment_image(gray, method, params, model=None):
    if method == "Hough circles":
        return segment_hough(gray, params)

    mask = segment_mask(gray, method, params, model)
    return extract_contours(mask), mask


def segment_mask(gray, method, params, model=None):
    blur_size = normalized_odd(params["blur"])
    morph_size = normalized_odd(params["morph_kernel"])
    morph_iter = int(params["morph_iter"])

    if blur_size > 1:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    if method == "Otsu + morphology":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "Manual threshold":
        _, mask = cv2.threshold(gray, int(params["threshold"]), 255, cv2.THRESH_BINARY)
    elif method == "Adaptive threshold":
        block = max(3, normalized_odd(params["adaptive_block"]))
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            int(params["adaptive_c"]),
        )
    elif method == "Canny + contours":
        low = int(params["threshold"])
        high = min(255, max(low + 1, low * 2))
        edges = cv2.Canny(gray, low, high)
        mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((morph_size, morph_size), np.uint8), iterations=1)
        return mask
    elif method == "Watershed":
        return segment_watershed(gray, params)
    elif method == U_NET_METHOD:
        return segment_unet(gray, params, model)
    else:
        raise ValueError(f"Unknown method: {method}")

    mask = choose_foreground(mask, gray)
    if morph_size > 1 and morph_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    return mask


def segment_unet(gray, params, model):
    if model is None:
        return np.zeros_like(gray)

    prob = predict_unet_multiscale(gray, model)
    threshold = float(params["unet_threshold"])
    mask = (prob >= threshold).astype(np.uint8) * 255

    morph_size = normalized_odd(params["morph_kernel"])
    if morph_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        iterations = max(1, int(params["morph_iter"]))
        mask = cv2.medianBlur(mask, morph_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    mask = split_touching_particles(mask, gray)
    return mask


def split_touching_particles(mask, gray):
    if np.count_nonzero(mask) == 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if np.count_nonzero(cleaned) == 0:
        return mask

    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    max_dist = float(dist.max())
    if max_dist <= 0:
        return cleaned

    _, sure_fg = cv2.threshold(dist, 0.35 * max_dist, 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(cleaned, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    marker_count, markers = cv2.connectedComponents(sure_fg)
    if marker_count <= 2:
        return cleaned

    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)

    separated = np.zeros_like(mask, dtype=np.uint8)
    separated[markers > 1] = 255
    return separated


def predict_unet_multiscale(gray, model, scales=(1.0, 2.0)):
    h, w = gray.shape
    probs = []
    for scale in scales:
        if scale == 1.0:
            scaled = gray
        else:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_prob = predict_unet_tiled(scaled, model)
        if scale != 1.0:
            scaled_prob = cv2.resize(scaled_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        probs.append(scaled_prob)
    return np.max(np.stack(probs, axis=0), axis=0)


def predict_unet_tiled(gray, model, tile_size=256, stride=128):
    h, w = gray.shape
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    padded = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape

    y_positions = list(range(0, max(1, ph - tile_size + 1), stride))
    x_positions = list(range(0, max(1, pw - tile_size + 1), stride))
    if y_positions[-1] != ph - tile_size:
        y_positions.append(ph - tile_size)
    if x_positions[-1] != pw - tile_size:
        x_positions.append(pw - tile_size)

    prob_sum = np.zeros((ph, pw), dtype=np.float32)
    prob_count = np.zeros((ph, pw), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                tile = padded[y : y + tile_size, x : x + tile_size].astype(np.float32) / 255.0
                img_tensor = torch.tensor(tile, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                logits = model(img_tensor)
                tile_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                prob_sum[y : y + tile_size, x : x + tile_size] += tile_prob
                prob_count[y : y + tile_size, x : x + tile_size] += 1.0

    prob = prob_sum / np.maximum(prob_count, 1e-6)
    return prob[:h, :w]


def segment_watershed(gray, params):
    blur_size = normalized_odd(params["blur"])
    morph_size = normalized_odd(params["morph_kernel"])
    morph_iter = int(params["morph_iter"])

    blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = choose_foreground(binary, gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=max(1, morph_iter))
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255
    return mask


def segment_hough(gray, params):
    blur_size = normalized_odd(params["blur"])
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    min_radius = int(params["min_radius"])
    max_radius = max(min_radius + 1, int(params["max_radius"]))
    min_dist = max(2, min_radius * 2)
    circles = None

    try:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT_ALT,
            dp=1.5,
            minDist=min_dist,
            param1=300,
            param2=float(params["hough_param2"]),
            minRadius=min_radius,
            maxRadius=max_radius,
        )
    except cv2.error:
        pass

    if circles is None:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=max(20, int(params["threshold"])),
            param2=max(5, int(float(params["hough_param2"]) * 50)),
            minRadius=min_radius,
            maxRadius=max_radius,
        )

    mask = np.zeros_like(gray, dtype=np.uint8)
    contours = []
    if circles is None:
        return contours, mask

    kept = nms_circles(circles[0], overlap=0.45)
    for cx, cy, cr in kept:
        cx, cy, cr = int(cx), int(cy), int(cr)
        if cr <= 0:
            continue
        cv2.circle(mask, (cx, cy), cr, 255, -1)
        angles = np.linspace(0, 2 * np.pi, max(36, cr * 2), endpoint=False)
        pts = np.array([[[int(cx + cr * np.cos(a)), int(cy + cr * np.sin(a))]] for a in angles], dtype=np.int32)
        contours.append(pts)
    return contours, mask


def nms_circles(circles, overlap=0.45):
    ordered = sorted(circles, key=lambda c: c[2], reverse=True)
    kept = []
    for cx, cy, cr in ordered:
        keep = True
        for kx, ky, kr in kept:
            dist = np.hypot(cx - kx, cy - ky)
            if dist < min(cr, kr) * (2 - overlap):
                keep = False
                break
        if keep:
            kept.append((cx, cy, cr))
    return kept

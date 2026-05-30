import cv2
import numpy as np

from segmentation import U_NET_METHOD


def measure_particles(contours, image_shape, params, method=None):
    particles = []
    if method == U_NET_METHOD:
        min_area = max(float(params["min_area"]), 80.0)
        min_circularity = max(float(params["circularity"]), 0.10)
    else:
        min_area = float(params["min_area"])
        min_circularity = float(params["circularity"])
    max_area = float(params["max_area"])
    h, w = image_shape[:2]
    roi_limit_y = int(h * 0.94)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if y + bh / 2 > roi_limit_y:
            continue
        if x < 0 or y < 0 or x + bw > w + 2 or y + bh > h + 2:
            continue

        eq_diameter = 2 * np.sqrt(area / np.pi)
        particles.append(
            {
                "contour": cnt,
                "area": area,
                "diameter": eq_diameter,
                "circularity": circularity,
                "bbox": (x, y, bw, bh),
            }
        )

    particles.sort(key=lambda p: (p["bbox"][1], p["bbox"][0]))
    return particles


def get_approach_type(method):
    if method == U_NET_METHOD:
        return "Нейромережевий підхід"
    return "Класичний алгоритм OpenCV"


def build_summary(method, particles, analysis_mask, image_shape):
    if particles:
        areas = np.array([p["area"] for p in particles], dtype=np.float32)
        diameters = np.array([p["diameter"] for p in particles], dtype=np.float32)
        mask_area = np.count_nonzero(analysis_mask) if analysis_mask is not None else 0
        image_area = image_shape[0] * image_shape[1]
        return (
            f"Метод: {method}\n"
            f"Тип: {get_approach_type(method)}\n"
            f"Кількість частинок: {len(particles)}\n"
            f"Середня площа: {areas.mean():.1f} px; медіана: {np.median(areas):.1f} px\n"
            f"Середній еквівалентний діаметр: {diameters.mean():.1f} px\n"
            f"Покриття маскою: {(mask_area / image_area) * 100:.2f}%"
        )

    return (
        f"Метод: {method}\n"
        f"Тип: {get_approach_type(method)}\n"
        f"Частинки не знайдено. {get_empty_result_hint(method)}"
    )


def get_empty_result_hint(method):
    if method == "Otsu + morphology":
        return "Спробуйте змінити згладжування або морфологію."
    if method == "Adaptive threshold":
        return "Спробуйте змінити Adaptive block або Adaptive C."
    if method == "Hough circles":
        return "Спробуйте зменшити Hough чутливість або розширити діапазон радіусів."
    if method == U_NET_METHOD:
        return "Спробуйте зменшити U-Net поріг, додати більше ручної розмітки або повторити навчання."
    return "Спробуйте змінити параметри або вибрати інший метод."

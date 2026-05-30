import csv
import json
import os
from tkinter import filedialog, messagebox

import cv2
import numpy as np


def save_markup(app):
    if app.cv_image is None:
        messagebox.showwarning("Немає зображення", "Спочатку відкрийте SEM-знімок.")
        return
    if not app.all_polygons:
        messagebox.showwarning("Немає розмітки", "Спочатку обведіть хоча б одну частинку.")
        return

    default_name = "sem_markup.json"
    if app.file_path:
        base_name = os.path.splitext(os.path.basename(app.file_path))[0]
        default_name = f"{base_name}_markup.json"

    file_path = filedialog.asksaveasfilename(
        title="Зберегти ручну розмітку",
        defaultextension=".json",
        initialfile=default_name,
        filetypes=[("JSON markup", "*.json"), ("All files", "*.*")],
    )
    if not file_path:
        return

    h, w = app.cv_image.shape[:2]
    data = {
        "image_path": app.file_path,
        "image_width": w,
        "image_height": h,
        "polygon_count": len(app.all_polygons),
        "polygons": [[[int(x), int(y)] for x, y in polygon] for polygon in app.all_polygons],
    }

    with open(file_path, "w", encoding="utf-8") as markup_file:
        json.dump(data, markup_file, ensure_ascii=False, indent=2)

    app.status_var.set(f"Розмітку збережено: {len(app.all_polygons)} полігонів")
    messagebox.showinfo("Збережено", f"Розмітку збережено:\n{file_path}")


def load_markup(app):
    if app.cv_image is None:
        messagebox.showwarning(
            "Немає зображення",
            "Спочатку відкрийте SEM-знімок, до якого належить розмітка.",
        )
        return

    file_path = filedialog.askopenfilename(
        title="Завантажити ручну розмітку",
        filetypes=[("JSON markup", "*.json"), ("All files", "*.*")],
    )
    if not file_path:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as markup_file:
            data = json.load(markup_file)
    except Exception as exc:
        messagebox.showerror("Помилка", f"Не вдалося прочитати розмітку:\n{exc}")
        return

    h, w = app.cv_image.shape[:2]
    saved_w = int(data.get("image_width", -1))
    saved_h = int(data.get("image_height", -1))
    if saved_w != w or saved_h != h:
        proceed = messagebox.askyesno(
            "Інший розмір зображення",
            f"Розмітка збережена для {saved_w}x{saved_h}, а поточне зображення {w}x{h}.\n"
            "Завантажити все одно?",
        )
        if not proceed:
            return

    polygons = []
    for polygon in data.get("polygons", []):
        clean_polygon = []
        for point in polygon:
            if len(point) >= 2:
                clean_polygon.append((int(point[0]), int(point[1])))
        if len(clean_polygon) >= 3:
            polygons.append(clean_polygon)

    if not polygons:
        messagebox.showwarning("Порожня розмітка", "У файлі немає коректних полігонів.")
        return

    app.current_poly_points = []
    app.all_polygons = polygons
    app.manual_mask_gt = None
    app.clear_results()
    app.update_canvas_view()
    app.status_var.set(f"Розмітку завантажено: {len(app.all_polygons)} полігонів")


def save_results(app):
    if app.cv_image is None:
        messagebox.showwarning("Немає зображення", "Спочатку відкрийте SEM-знімок.")
        return
    if not app.last_particles:
        messagebox.showwarning(
            "Немає результатів",
            "Спочатку запустіть аналіз, щоб отримати контури частинок.",
        )
        return

    file_path = filedialog.asksaveasfilename(
        title="Зберегти результат аналізу",
        defaultextension=".png",
        filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
    )
    if not file_path:
        return

    annotated = build_annotated_image(app)
    ext = os.path.splitext(file_path)[1].lower() or ".png"
    if ext not in [".png", ".jpg", ".jpeg"]:
        file_path += ".png"
        ext = ".png"

    ok, buffer = cv2.imencode(ext, annotated)
    if not ok:
        messagebox.showerror("Помилка", "Не вдалося підготувати зображення для збереження.")
        return
    buffer.tofile(file_path)

    csv_path = os.path.splitext(file_path)[0] + "_measurements.csv"
    save_measurements_csv(app, csv_path)
    plot_path = os.path.splitext(file_path)[0] + "_diameter_distribution.png"
    plot_saved = save_size_distribution_plot(csv_path, plot_path)

    if app.analysis_mask is not None:
        mask_path = os.path.splitext(file_path)[0] + "_mask.png"
        ok, mask_buffer = cv2.imencode(".png", app.analysis_mask)
        if ok:
            mask_buffer.tofile(mask_path)

    plot_message = f"\n\nГрафік розподілу діаметрів:\n{plot_path}" if plot_saved else ""
    messagebox.showinfo(
        "Збережено",
        f"Зображення з контурами:\n{file_path}\n\nТаблиця вимірювань:\n{csv_path}{plot_message}",
    )


def build_annotated_image(app):
    annotated = app.cv_image.copy()
    for idx, particle in enumerate(app.last_particles, start=1):
        contour = particle["contour"]
        x, y, _, _ = particle["bbox"]
        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
        cv2.putText(
            annotated,
            str(idx),
            (x + 4, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def save_measurements_csv(app, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["method", app.algorithm_var.get()])
        writer.writerow(["id", "area_px", "equivalent_diameter_px", "circularity", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        for idx, particle in enumerate(app.last_particles, start=1):
            x, y, w, h = particle["bbox"]
            writer.writerow(
                [
                    idx,
                    f"{particle['area']:.2f}",
                    f"{particle['diameter']:.2f}",
                    f"{particle['circularity']:.4f}",
                    x,
                    y,
                    w,
                    h,
                ]
            )


def save_size_distribution_plot(csv_path, plot_path):
    diameters = read_equivalent_diameters(csv_path)
    if not diameters:
        return False

    if save_matplotlib_distribution_plot(diameters, plot_path):
        return True

    return save_opencv_distribution_plot(diameters, plot_path)


def save_matplotlib_distribution_plot(diameters, plot_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    values = np.array(diameters, dtype=np.float32)
    bin_count = min(18, max(6, int(np.sqrt(len(values))) + 1))
    mean_value = float(values.mean())
    median_value = float(np.median(values))

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=140)
    ax.hist(
        values,
        bins=bin_count,
        color="#2f80ed",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.92,
    )
    ax.axvline(mean_value, color="#d9480f", linewidth=2, label=f"Mean: {mean_value:.2f} px")
    ax.axvline(median_value, color="#2b8a3e", linewidth=2, linestyle="--", label=f"Median: {median_value:.2f} px")

    ax.set_title("Particle Size Distribution", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Equivalent diameter, px", fontsize=12)
    ax.set_ylabel("Particle count", fontsize=12)
    ax.text(
        0.0,
        1.015,
        f"n = {len(values)}",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#555555",
        ha="left",
        va="bottom",
    )
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    try:
        fig.savefig(plot_path, bbox_inches="tight", facecolor="white")
        return True
    finally:
        plt.close(fig)


def save_opencv_distribution_plot(diameters, plot_path):
    width, height = 1000, 650
    left, right, top, bottom = 110, 55, 92, 105
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    values = np.array(diameters, dtype=np.float32)
    bin_count = min(18, max(6, int(np.sqrt(len(values))) + 1))
    hist, edges = np.histogram(values, bins=bin_count)
    max_count = max(int(hist.max()), 1)

    axis_color = (55, 55, 55)
    grid_color = (226, 226, 226)
    bar_color = (237, 128, 47)
    bar_border = (160, 77, 15)
    text_color = (30, 30, 30)

    x0, y0 = left, height - bottom
    x1, y1 = width - right, top
    cv2.line(image, (x0, y0), (x1, y0), axis_color, 2)
    cv2.line(image, (x0, y0), (x0, y1), axis_color, 2)

    for tick in range(0, 6):
        y = int(y0 - tick * plot_h / 5)
        value = int(round(tick * max_count / 5))
        cv2.line(image, (x0, y), (x1, y), grid_color, 1)
        cv2.putText(image, str(value), (45, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

    gap = max(3, int(plot_w / bin_count * 0.10))
    for idx, count in enumerate(hist):
        bar_left = int(x0 + idx * plot_w / bin_count) + gap
        bar_right = int(x0 + (idx + 1) * plot_w / bin_count) - gap
        bar_top = int(y0 - (int(count) / max_count) * plot_h)
        cv2.rectangle(image, (bar_left, bar_top), (bar_right, y0), bar_color, -1)
        cv2.rectangle(image, (bar_left, bar_top), (bar_right, y0), bar_border, 1)

    label_step = max(1, bin_count // 7)
    for idx in range(0, bin_count + 1, label_step):
        x = int(x0 + idx * plot_w / bin_count)
        cv2.line(image, (x, y0), (x, y0 + 6), axis_color, 1)
        label = f"{edges[idx]:.1f}"
        cv2.putText(image, label, (x - 22, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

    mean_value = float(values.mean())
    median_value = float(np.median(values))
    for value, color in [(mean_value, (15, 72, 217)), (median_value, (62, 138, 43))]:
        x = int(x0 + (value - edges[0]) / max(edges[-1] - edges[0], 1e-6) * plot_w)
        cv2.line(image, (x, y0), (x, y1), color, 2)

    title = "Particle Size Distribution"
    subtitle = f"n={len(values)}, mean={mean_value:.2f}px, median={median_value:.2f}px"
    cv2.putText(image, title, (left, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)
    cv2.putText(image, subtitle, (left, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(image, "Equivalent diameter, px", (width // 2 - 150, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
    cv2.putText(image, "Particle count", (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
    cv2.putText(image, "mean", (x1 - 130, top - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 72, 217), 1, cv2.LINE_AA)
    cv2.putText(image, "median", (x1 - 130, top - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (62, 138, 43), 1, cv2.LINE_AA)

    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return False
    buffer.tofile(plot_path)
    return True


def read_equivalent_diameters(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.reader(csv_file))

    header_index = None
    diameter_index = None
    for row_index, row in enumerate(rows):
        if "equivalent_diameter_px" in row:
            header_index = row_index
            diameter_index = row.index("equivalent_diameter_px")
            break

    if header_index is None:
        return []

    diameters = []
    for row in rows[header_index + 1:]:
        if len(row) <= diameter_index:
            continue
        try:
            diameters.append(float(row[diameter_index]))
        except ValueError:
            continue
    return diameters


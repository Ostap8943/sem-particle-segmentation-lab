import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from file_operations import load_markup as load_markup_file
from file_operations import save_markup as save_markup_file
from file_operations import save_results as save_results_file
from particle_metrics import build_summary, measure_particles
from segmentation import U_NET_METHOD, segment_image
from training_utils import create_manual_mask, train_unet
from ui_helpers import add_checkbox, add_combo, add_slider, make_button, make_menu_button
from unet_model import UNet


class AutomatedParticleApp:
    def __init__(self, window):
        self.window = window
        self.window.title("SEM Particle Segmentation Lab")
        self.window.geometry("1450x850")
        self.window.configure(bg="#263238")

        self.cv_image = None
        self.file_path = None
        self.model = UNet()
        self.is_trained = False
        self.manual_mask_gt = None

        self.current_poly_points = []
        self.all_polygons = []
        self.auto_contours = []
        self.analysis_mask = None
        self.last_particles = []
        self.training_run_id = 0
        self.last_training_info = None

        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.current_total_scale = 1.0
        self.interaction_mode = tk.StringVar(value="pan")

        self.algorithm_var = tk.StringVar(value="Hough circles")
        self.status_var = tk.StringVar(value="Відкрийте SEM-знімок для аналізу")

        self.blur_var = tk.IntVar(value=5)
        self.threshold_var = tk.IntVar(value=128)
        self.adaptive_block_var = tk.IntVar(value=31)
        self.adaptive_c_var = tk.IntVar(value=4)
        self.morph_kernel_var = tk.IntVar(value=5)
        self.morph_iter_var = tk.IntVar(value=1)
        self.min_area_var = tk.IntVar(value=20)
        self.max_area_var = tk.IntVar(value=12000)
        self.circularity_var = tk.DoubleVar(value=0.60)
        self.hough_param2_var = tk.DoubleVar(value=0.88)
        self.min_radius_var = tk.IntVar(value=10)
        self.max_radius_var = tk.IntVar(value=80)
        self.unet_threshold_var = tk.DoubleVar(value=0.25)
        self.epochs_var = tk.IntVar(value=40)
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        self.use_augmentation_var = tk.BooleanVar(value=True)

        self.build_ui()

    def build_ui(self):
        top_panel = tk.Frame(self.window, bg="#37474f", height=64)
        top_panel.pack(fill=tk.X, side=tk.TOP)

        make_button(top_panel, "Відкрити SEM", self.load_image, "#2e7d32").pack(side=tk.LEFT, padx=10, pady=10)
        make_menu_button(
            top_panel,
            "Розмітка",
            [
                ("Скинути розмітку", self.clear_markup),
                ("Зберегти розмітку", self.save_markup),
                ("Завантажити розмітку", self.load_markup),
            ],
        ).pack(side=tk.LEFT, padx=4, pady=10)
        self.draw_mode_button = make_button(top_panel, "Малювати", lambda: self.set_interaction_mode("draw"), "#00897b")
        self.draw_mode_button.pack(side=tk.LEFT, padx=4, pady=10)
        self.pan_mode_button = make_button(top_panel, "Рух фото", lambda: self.set_interaction_mode("pan"), "#546e7a")
        self.pan_mode_button.pack(side=tk.LEFT, padx=4, pady=10)
        make_button(top_panel, "Навчити U-Net", self.train_network, "#ef6c00").pack(side=tk.LEFT, padx=4, pady=10)
        make_button(top_panel, "Запустити аналіз", self.analyze_image, "#1565c0").pack(side=tk.LEFT, padx=4, pady=10)
        make_button(top_panel, "Зберегти результат", self.save_results, "#6a1b9a").pack(side=tk.LEFT, padx=4, pady=10)

        ttk.Label(top_panel, textvariable=self.status_var, background="#37474f", foreground="white").pack(
            side=tk.LEFT, padx=18
        )

        main_frame = tk.Frame(self.window, bg="#263238")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left_frame = tk.Frame(main_frame, bg="#263238")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_frame = tk.LabelFrame(
            left_frame,
            text=" Перегляд і ручна розмітка: лівий клік - точки полігона, правий клік - замкнути, колесо - масштаб, середня кнопка - рух ",
            fg="white",
            bg="#263238",
            font=("Arial", 10, "bold"),
        )
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#1f2a30", bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.close_polygon)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.drag_pan)
        self.set_interaction_mode("pan")

        right_frame = tk.Frame(main_frame, bg="#263238", width=430)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        settings_frame = tk.LabelFrame(
            right_frame,
            text=" Керування алгоритмами ",
            fg="white",
            bg="#263238",
            font=("Arial", 10, "bold"),
        )
        settings_frame.pack(fill=tk.X, padx=5, pady=5)

        add_combo(
            settings_frame,
            "Підхід сегментації",
            self.algorithm_var,
            [
                "Otsu + morphology",
                "Adaptive threshold",
                "Hough circles",
                U_NET_METHOD,
            ],
        )

        self.params_frame = tk.LabelFrame(
            settings_frame,
            text=" Параметри вибраного підходу ",
            fg="white",
            bg="#263238",
            font=("Arial", 9, "bold"),
        )
        self.params_frame.pack(fill=tk.X, padx=8, pady=6)

        self.param_rows = {
            "blur": add_slider(self.params_frame, "Згладжування", self.blur_var, 1, 21, 2),
            "threshold": add_slider(self.params_frame, "Поріг", self.threshold_var, 0, 255, 1),
            "canny": add_slider(self.params_frame, "Canny поріг", self.threshold_var, 0, 255, 1),
            "min_area": add_slider(self.params_frame, "Мін. площа, px", self.min_area_var, 5, 3000, 5),
            "max_area": add_slider(self.params_frame, "Макс. площа, px", self.max_area_var, 100, 100000, 100),
            "circularity": add_slider(self.params_frame, "Мін. круглість", self.circularity_var, 0.0, 1.0, 0.01),
            "morph": add_slider(self.params_frame, "Морфологія", self.morph_kernel_var, 1, 31, 2),
            "morph_iter": add_slider(self.params_frame, "Ітерації морф.", self.morph_iter_var, 0, 5, 1),
            "adaptive_block": add_slider(self.params_frame, "Adaptive block", self.adaptive_block_var, 3, 99, 2),
            "adaptive_c": add_slider(self.params_frame, "Adaptive C", self.adaptive_c_var, -20, 20, 1),
            "hough_sensitivity": add_slider(self.params_frame, "Hough чутливість", self.hough_param2_var, 0.10, 0.99, 0.01),
            "min_radius": add_slider(self.params_frame, "Мін. радіус", self.min_radius_var, 1, 120, 1),
            "max_radius": add_slider(self.params_frame, "Макс. радіус", self.max_radius_var, 5, 220, 1),
            "unet_threshold": add_slider(self.params_frame, "U-Net поріг", self.unet_threshold_var, 0.05, 0.60, 0.01),
            "epochs": add_slider(self.params_frame, "Епохи", self.epochs_var, 10, 120, 10),
            "learning_rate": add_slider(self.params_frame, "Learning rate", self.learning_rate_var, 0.0001, 0.01, 0.0001),
            "augmentation": add_checkbox(self.params_frame, "Data augmentation", self.use_augmentation_var),
        }
        self.sem_preset_button = make_button(
            self.params_frame, "Профіль SEM-кульки", self.apply_sem_spheres_preset, "#455a64"
        )
        self.algorithm_var.trace_add("write", lambda *_: self.on_algorithm_changed())
        self.update_visible_params()

        table_frame = tk.LabelFrame(
            right_frame,
            text=" Результати розпізнавання ",
            fg="white",
            bg="#263238",
            font=("Arial", 10, "bold"),
        )
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        cols = ("ID", "Area", "Diameter", "Circularity", "BBox")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=18)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#eceff1", fieldbackground="#eceff1", rowheight=24)
        style.configure("Treeview.Heading", font=("Arial", 9, "bold"))

        headings = {
            "ID": ("ID", 46),
            "Area": ("Площа", 74),
            "Diameter": ("Діаметр", 74),
            "Circularity": ("Круглість", 74),
            "BBox": ("BBox", 150),
        }
        for col, (text, width) in headings.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        summary_frame = tk.LabelFrame(
            right_frame,
            text=" Статистика ",
            fg="white",
            bg="#263238",
            font=("Arial", 10, "bold"),
        )
        summary_frame.pack(fill=tk.X, padx=5, pady=5)

        self.summary_text = tk.Text(summary_frame, height=7, bg="#eceff1", fg="#102027", wrap=tk.WORD)
        self.summary_text.pack(fill=tk.X, padx=5, pady=5)
        self.summary_text.insert(tk.END, "Після аналізу тут буде кількість частинок, площа, діаметр і покриття маскою.")
        self.summary_text.configure(state=tk.DISABLED)

    def set_interaction_mode(self, mode):
        self.interaction_mode.set(mode)
        if hasattr(self, "draw_mode_button"):
            if mode == "draw":
                self.draw_mode_button.configure(bg="#00897b", relief=tk.SUNKEN)
                self.pan_mode_button.configure(bg="#546e7a", relief=tk.FLAT)
                self.canvas_frame.configure(
                    text=" Перегляд і ручна розмітка: режим Малювати - лівий клік додає точки, правий клік замикає полігон, колесо масштабує біля курсора "
                )
            else:
                self.draw_mode_button.configure(bg="#546e7a", relief=tk.FLAT)
                self.pan_mode_button.configure(bg="#00897b", relief=tk.SUNKEN)
                self.canvas_frame.configure(
                    text=" Перегляд і ручна розмітка: режим Рух фото - затисніть ліву кнопку і перетягуйте зображення, колесо масштабує біля курсора "
                )

    def update_visible_params(self):
        if not hasattr(self, "param_rows"):
            return

        method_params = {
            "Otsu + morphology": ["blur", "min_area", "morph"],
            "Adaptive threshold": ["adaptive_block", "adaptive_c", "min_area"],
            "Hough circles": ["hough_sensitivity", "min_radius", "max_radius"],
            U_NET_METHOD: ["unet_threshold", "epochs", "augmentation"],
        }

        visible_keys = method_params.get(self.algorithm_var.get(), [])
        for row in self.param_rows.values():
            row.pack_forget()
        for key in visible_keys:
            self.param_rows[key].pack(fill=tk.X, padx=8, pady=2)

        self.sem_preset_button.pack_forget()
        if self.algorithm_var.get() == "Hough circles":
            self.sem_preset_button.pack(fill=tk.X, padx=8, pady=5)

    def on_algorithm_changed(self):
        self.apply_recommended_params(self.algorithm_var.get())
        self.update_visible_params()

    def apply_recommended_params(self, method):
        if method == "Otsu + morphology":
            self.blur_var.set(3)
            self.min_area_var.set(20)
            self.max_area_var.set(12000)
            self.circularity_var.set(0.60)
            self.morph_kernel_var.set(1)
            self.morph_iter_var.set(1)
        elif method == "Adaptive threshold":
            self.adaptive_block_var.set(31)
            self.adaptive_c_var.set(-5)
            self.min_area_var.set(20)
            self.max_area_var.set(12000)
            self.circularity_var.set(0.60)
            self.morph_kernel_var.set(1)
            self.morph_iter_var.set(1)
        elif method == "Hough circles":
            self.set_sem_spheres_params()
        elif method == U_NET_METHOD:
            self.unet_threshold_var.set(0.25)
            self.epochs_var.set(40)
            self.morph_kernel_var.set(3)
            self.morph_iter_var.set(1)

    def apply_sem_spheres_preset(self, update_status=True):
        self.algorithm_var.set("Hough circles")
        self.set_sem_spheres_params()
        if update_status:
            self.status_var.set("Профіль SEM-кульки застосовано. Натисніть 'Запустити аналіз'.")

    def set_sem_spheres_params(self):
        self.blur_var.set(5)
        self.threshold_var.set(128)
        self.min_area_var.set(120)
        self.max_area_var.set(12000)
        self.circularity_var.set(0.60)
        self.morph_kernel_var.set(3)
        self.morph_iter_var.set(1)
        self.hough_param2_var.set(0.88)
        self.min_radius_var.set(10)
        self.max_radius_var.set(80)

    def get_analysis_params(self):
        params = {
            "blur": self.blur_var.get(),
            "threshold": self.threshold_var.get(),
            "adaptive_block": self.adaptive_block_var.get(),
            "adaptive_c": self.adaptive_c_var.get(),
            "morph_kernel": self.morph_kernel_var.get(),
            "morph_iter": self.morph_iter_var.get(),
            "min_area": self.min_area_var.get(),
            "max_area": self.max_area_var.get(),
            "circularity": self.circularity_var.get(),
            "hough_param2": self.hough_param2_var.get(),
            "min_radius": self.min_radius_var.get(),
            "max_radius": self.max_radius_var.get(),
            "unet_threshold": self.unet_threshold_var.get(),
        }
        if self.algorithm_var.get() == U_NET_METHOD:
            params["morph_kernel"] = 3
            params["morph_iter"] = 1
            params["min_area"] = 80
            params["circularity"] = 0.10
        return params

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff")])
        if not file_path:
            return

        try:
            nparr = np.fromfile(file_path, np.uint8)
            self.cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as exc:
            messagebox.showerror("Помилка", f"Не вдалося прочитати файл: {exc}")
            return

        if self.cv_image is None:
            messagebox.showerror("Помилка", "OpenCV не зміг декодувати зображення.")
            return

        self.file_path = file_path
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.clear_results()
        self.update_canvas_view()
        h, w = self.cv_image.shape[:2]
        self.status_var.set(f"Зображення: {w}x{h}px")

    def update_canvas_view(self):
        if self.cv_image is None:
            return

        img_src = self.cv_image.copy()
        h, w = img_src.shape[:2]
        base_scale = min(820 / w, 700 / h)
        total_scale = base_scale * self.zoom_level
        new_w, new_h = int(w * total_scale), int(h * total_scale)
        if new_w <= 0 or new_h <= 0:
            return

        resized = cv2.resize(img_src, (new_w, new_h))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(rgb_img)
        self.tk_img = ImageTk.PhotoImage(image=self.pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_img, tags="img_tag")
        self.current_total_scale = total_scale
        self.redraw_markup_elements()
        self.redraw_auto_contours()

    def get_image_base_scale(self):
        if self.cv_image is None:
            return 1.0
        h, w = self.cv_image.shape[:2]
        return min(820 / w, 700 / h)

    def redraw_auto_contours(self):
        for idx, cnt in enumerate(self.auto_contours):
            screen_pts = []
            for pt in cnt:
                ox, oy = pt[0][0], pt[0][1]
                sx = int(ox * self.current_total_scale) + self.pan_x
                sy = int(oy * self.current_total_scale) + self.pan_y
                screen_pts.extend([sx, sy])

            if len(screen_pts) > 5:
                self.canvas.create_polygon(screen_pts, outline="#00e676", fill="", width=2, tags="auto_contour")
                x, y, _, _ = cv2.boundingRect(cnt)
                sx_text = int(x * self.current_total_scale) + self.pan_x
                sy_text = int(y * self.current_total_scale) + self.pan_y
                self.canvas.create_text(
                    sx_text + 12,
                    sy_text + 12,
                    text=str(idx + 1),
                    fill="#ff1744",
                    font=("Arial", 9, "bold"),
                    tags="auto_text",
                )

    def on_mouse_wheel(self, event):
        if self.cv_image is None:
            return

        old_scale = self.current_total_scale
        image_x = (event.x - self.pan_x) / old_scale
        image_y = (event.y - self.pan_y) / old_scale

        self.zoom_level = max(0.25, self.zoom_level - 0.1) if event.delta < 0 else min(6.0, self.zoom_level + 0.1)
        new_scale = self.get_image_base_scale() * self.zoom_level
        self.pan_x = int(event.x - image_x * new_scale)
        self.pan_y = int(event.y - image_y * new_scale)
        self.update_canvas_view()

    def start_pan(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def drag_pan(self, event):
        if self.cv_image is None:
            return
        self.pan_x += event.x - self.drag_start_x
        self.pan_y += event.y - self.drag_start_y
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.update_canvas_view()

    def on_left_press(self, event):
        if self.interaction_mode.get() == "pan":
            self.start_pan(event)

    def on_left_drag(self, event):
        if self.interaction_mode.get() == "pan":
            self.drag_pan(event)

    def on_left_release(self, event):
        if self.interaction_mode.get() == "draw":
            self.on_canvas_click(event)

    def on_canvas_click(self, event):
        if self.cv_image is None:
            return
        orig_x = int((event.x - self.pan_x) / self.current_total_scale)
        orig_y = int((event.y - self.pan_y) / self.current_total_scale)
        h, w = self.cv_image.shape[:2]
        if 0 <= orig_x < w and 0 <= orig_y < h:
            self.current_poly_points.append((orig_x, orig_y))
            self.update_canvas_view()

    def close_polygon(self, _event):
        if self.interaction_mode.get() != "draw":
            return
        if len(self.current_poly_points) >= 3:
            self.all_polygons.append(self.current_poly_points)
            self.current_poly_points = []
            self.update_canvas_view()

    def redraw_markup_elements(self):
        for poly in self.all_polygons:
            screen_pts = []
            for ox, oy in poly:
                sx = int(ox * self.current_total_scale) + self.pan_x
                sy = int(oy * self.current_total_scale) + self.pan_y
                screen_pts.extend([sx, sy])
            if len(screen_pts) > 5:
                self.canvas.create_polygon(screen_pts, outline="#29b6f6", fill="", width=2, tags="polygon")

        for i, (ox, oy) in enumerate(self.current_poly_points):
            sx = int(ox * self.current_total_scale) + self.pan_x
            sy = int(oy * self.current_total_scale) + self.pan_y
            self.canvas.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill="#ff5252", tags="polygon")
            if i > 0:
                px, py = self.current_poly_points[i - 1]
                sx_prev = int(px * self.current_total_scale) + self.pan_x
                sy_prev = int(py * self.current_total_scale) + self.pan_y
                self.canvas.create_line(sx_prev, sy_prev, sx, sy, fill="#ff5252", width=2, tags="polygon")

    def clear_markup(self):
        self.current_poly_points = []
        self.all_polygons = []
        self.manual_mask_gt = None
        self.clear_results()
        self.update_canvas_view()

    def save_markup(self):
        save_markup_file(self)

    def load_markup(self):
        load_markup_file(self)

    def clear_results(self):
        self.auto_contours = []
        self.analysis_mask = None
        self.last_particles = []
        if hasattr(self, "tree"):
            for item in self.tree.get_children():
                self.tree.delete(item)
        self.set_summary("Результати очищено.")

    def set_summary(self, text):
        if not hasattr(self, "summary_text"):
            return
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, text)
        self.summary_text.configure(state=tk.DISABLED)

    def train_network(self):
        if self.cv_image is None:
            messagebox.showwarning("Немає зображення", "Спочатку відкрийте SEM-знімок.")
            return
        if not self.all_polygons:
            messagebox.showwarning("Розмітка", "Обведіть хоча б одну частинку полігоном.")
            return
        if len(self.all_polygons) < 10:
            proceed = messagebox.askyesno(
                "Мало навчальних прикладів",
                "Для U-Net бажано розмітити хоча б 10-20 частинок замкненими полігонами. "
                "Якщо навчати на меншій кількості, мережа може знаходити лише кілька схожих об'єктів. Продовжити?",
            )
            if not proceed:
                return

        aug_label = "з augmentation" if self.use_augmentation_var.get() else "без augmentation"
        self.manual_mask_gt = create_manual_mask(self.cv_image.shape[:2], self.all_polygons)
        requested_epochs = int(self.epochs_var.get())
        self.training_run_id += 1
        self.clear_results()
        self.status_var.set(
            f"U-Net запуск #{self.training_run_id}: готує патчі з {len(self.all_polygons)} полігонів, "
            f"епох={requested_epochs} ({aug_label})..."
        )
        self.window.update_idletasks()
        self.model = UNet()
        self.is_trained = False

        def update_status(epoch, epochs, loss, sample_count):
            self.status_var.set(
                f"U-Net запуск #{self.training_run_id}: епоха {epoch + 1}/{epochs}, "
                f"зразків={sample_count}, loss={loss:.5f}"
            )
            self.window.update_idletasks()

        final_loss, sample_count = train_unet(
            self.model,
            self.cv_image,
            self.manual_mask_gt,
            self.all_polygons,
            requested_epochs,
            self.learning_rate_var.get(),
            self.use_augmentation_var.get(),
            status_callback=update_status,
        )

        self.is_trained = True
        self.last_training_info = {
            "run_id": self.training_run_id,
            "epochs": requested_epochs,
            "samples": sample_count,
            "loss": final_loss,
        }
        self.status_var.set(
            f"U-Net навчено #{self.training_run_id}: епох={requested_epochs}, "
            f"зразків={sample_count}, loss={final_loss:.5f}. Запустіть аналіз."
        )
        messagebox.showinfo(
            "Готово",
            f"U-Net навчено.\nЗапуск: #{self.training_run_id}\nЕпох: {requested_epochs}\n"
            f"Зразків: {sample_count}\nФінальна loss: {final_loss:.5f}",
        )

    def analyze_image(self):
        if self.cv_image is None:
            messagebox.showwarning("Немає зображення", "Спочатку відкрийте SEM-знімок.")
            return

        method = self.algorithm_var.get()
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

        if method == U_NET_METHOD and not self.is_trained:
            messagebox.showwarning("U-Net", "Спочатку навчіть U-Net на ручній розмітці.")
            return

        params = self.get_analysis_params()
        contours, mask = segment_image(gray, method, params, model=self.model)
        self.analysis_mask = mask
        particles = measure_particles(contours, self.cv_image.shape[:2], params, method=method)
        self.last_particles = particles
        self.fill_results(particles)
        self.auto_contours = [p["contour"] for p in particles]
        self.update_canvas_view()
        self.status_var.set(f"{method}: знайдено частинок {len(particles)}")

    def fill_results(self, particles):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for idx, p in enumerate(particles, start=1):
            x, y, w, h = p["bbox"]
            self.tree.insert(
                "",
                tk.END,
                values=(
                    f"#{idx}",
                    f"{p['area']:.1f}",
                    f"{p['diameter']:.1f}",
                    f"{p['circularity']:.2f}",
                    f"X:{x}, Y:{y}, W:{w}, H:{h}",
                ),
            )

        summary = build_summary(self.algorithm_var.get(), particles, self.analysis_mask, self.cv_image.shape[:2])
        self.set_summary(summary)

    def save_results(self):
        save_results_file(self)

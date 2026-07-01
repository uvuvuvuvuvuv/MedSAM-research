# -*- coding: utf-8 -*-
# <<< add BEGIN
import os
import sys

# 这里不强制设置 QT_QPA_PLATFORM，留给你用环境变量/命令行控制
# os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# 可选：消除 torch.load 的 FutureWarning（不影响功能）
import torch
_orig_torch_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", True)
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_load

# 可选：允许从命令行传 checkpoint（不传则走原来的默认路径）
_ckpt_arg = None
for i, a in enumerate(sys.argv):
    if a == "--checkpoint" and i + 1 < len(sys.argv):
        _ckpt_arg = sys.argv[i + 1]
        break
# <<< add END

import time
from PyQt5.QtGui import (
    QBrush,
    QPainter,
    QPen,
    QPixmap,
    QKeySequence,
    QColor,
    QImage,
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QShortcut,
)

import numpy as np
from skimage import transform, io
import torch
from torch.nn import functional as F
from PIL import Image
from segment_anything import sam_model_registry

# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"
# 优先使用命令行 --checkpoint，否则走默认路径
MedSAM_CKPT_PATH = _ckpt_arg or "work_dir/MedSAM/medsam_vit_b.pth"
MEDSAM_IMG_INPUT_SIZE = 1024

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (H, W)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


print("Loading MedSAM model, a sec.")
tic = time.perf_counter()

# set up model
medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH).to(device)
medsam_model.eval()

print(f"Done, took {time.perf_counter() - tic}")


def np2pixmap(np_img):
    """
    numpy (H, W, 3/4/1) -> QPixmap
    """
    import numpy as np

    # 灰度图 -> 3 通道
    if np_img.ndim == 2:
        np_img = np.stack([np_img] * 3, axis=-1)

    h, w, c = np_img.shape
    if c == 1:
        np_img = np.repeat(np_img, 3, axis=2)
        c = 3
    if c >= 3:
        np_img = np_img[:, :, :3]
    else:
        raise ValueError(f"np2pixmap expects at least 1 channel, got {np_img.shape}")

    # 转为连续 uint8
    np_img = np.ascontiguousarray(np_img, dtype=np.uint8)

    bytes_per_line = 3 * w
    qImg = QImage(
        np_img.tobytes(),
        w,
        h,
        bytes_per_line,
        QImage.Format_RGB888,
    )
    return QPixmap.fromImage(qImg)


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
    (192, 192, 192),
    (64, 64, 64),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 0, 127),
    (192, 0, 192),
]


class Window(QWidget):
    def __init__(self):
        super().__init__()

        # configs
        self.half_point_size = 5  # radius of bbox starting and ending points

        # app stats
        self.image_path = None
        self.color_idx = 0
        self.bg_img = None
        self.is_mouse_down = False
        self.rect = None
        self.point_size = self.half_point_size * 2
        self.start_point = None
        self.end_point = None
        self.start_pos = (None, None)
        self.embedding = None
        self.prev_mask = None

        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)

        # 布局：不要把 parent 传给 QVBoxLayout/QHBoxLayout，再调用 setLayout，否则会有 QLayout 警告
        vbox = QVBoxLayout()
        vbox.addWidget(self.view)

        load_button = QPushButton("Load Image")
        save_button = QPushButton("Save Mask")

        hbox = QHBoxLayout()
        hbox.addWidget(load_button)
        hbox.addWidget(save_button)

        vbox.addLayout(hbox)
        self.setLayout(vbox)

        # keyboard shortcuts
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(lambda: quit())

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)

        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_mask)

        # 启动时先让用户选一张图片（如果取消，不退出程序，只是提示）
        self.load_image(initial=True)

    def undo(self):
        if self.prev_mask is None:
            print("No previous mask record")
            return

        self.color_idx -= 1

        bg = Image.fromarray(self.img_3c.astype("uint8"))
        mask = Image.fromarray(self.prev_mask.astype("uint8"))
        img = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

        self.mask_c = self.prev_mask
        self.prev_mask = None

    def load_image(self, initial=False):
        file_path, file_type = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp)"
        )

        if not file_path:
            print("No image path specified, plz select an image")
            # 初次启动如果不选图，就不继续初始化；但程序仍然活着
            return

        img_np = io.imread(file_path)

        # ---- 统一到 3 通道 RGB，避免 4 通道 RGBA / 单通道报错 ----
        if img_np.ndim == 2:
            # 灰度图 H x W
            img_3c = np.stack([img_np] * 3, axis=-1)  # H x W x 3
        elif img_np.ndim == 3:
            h, w, c = img_np.shape
            if c == 3:
                # 正常 RGB
                img_3c = img_np
            elif c == 4:
                # RGBA，去掉 alpha
                img_3c = img_np[:, :, :3]
            elif c == 1:
                # 单通道，加到 3 通道
                img_3c = np.repeat(img_np, 3, axis=2)
            else:
                # 奇怪的多通道，先取前三个当 RGB
                img_3c = img_np[:, :, :3]
        else:
            raise ValueError(f"Unsupported image shape: {img_np.shape}")

        # 再保险：确保一定是 H x W x 3
        assert img_3c.ndim == 3 and img_3c.shape[2] == 3, f"Unexpected image shape after convert: {img_3c.shape}"

        self.img_3c = img_3c
        self.image_path = file_path
        self.get_embeddings()
        pixmap = np2pixmap(self.img_3c)

        H, W, _ = self.img_3c.shape

        self.scene = QGraphicsScene(0, 0, W, H)
        self.end_point = None
        self.rect = None
        self.bg_img = self.scene.addPixmap(pixmap)
        self.bg_img.setPos(0, 0)
        self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype="uint8")
        self.view.setScene(self.scene)

        # events
        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release

    def mouse_press(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = True
        self.start_pos = ev.scenePos().x(), ev.scenePos().y()
        self.start_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

    def mouse_move(self, ev):
        if not self.is_mouse_down:
            return

        x, y = ev.scenePos().x(), ev.scenePos().y()

        if self.end_point is not None:
            self.scene.removeItem(self.end_point)
        self.end_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

        if self.rect is not None:
            self.scene.removeItem(self.rect)
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)
        self.rect = self.scene.addRect(
            xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"))
        )

    def mouse_release(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)

        self.is_mouse_down = False

        H, W, _ = self.img_3c.shape
        box_np = np.array([[xmin, ymin, xmax, ymax]])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        sam_mask = medsam_inference(medsam_model, self.embedding, box_1024, H, W)

        self.prev_mask = self.mask_c.copy()
        self.mask_c[sam_mask != 0] = colors[self.color_idx % len(colors)]
        self.color_idx += 1

        bg = Image.fromarray(self.img_3c.astype("uint8"))
        mask = Image.fromarray(self.mask_c.astype("uint8"))
        img = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

    def save_mask(self):
        if not self.image_path:
            print("No image loaded, cannot save mask.")
            return
        out_path = f"{os.path.splitext(self.image_path)[0]}_mask.png"
        io.imsave(out_path, self.mask_c)
        print(f"Mask saved to: {out_path}")

    @torch.no_grad()
    def get_embeddings(self):
        print("Calculating embedding, gui may be unresponsive.")

        # 兜底再检查一下通道（按理说 load_image 已经保证是 3 通道）
        if self.img_3c.ndim == 2:
            self.img_3c = np.stack([self.img_3c] * 3, axis=-1)
        elif self.img_3c.ndim == 3 and self.img_3c.shape[2] != 3:
            if self.img_3c.shape[2] >= 3:
                self.img_3c = self.img_3c[:, :, :3]
            else:
                self.img_3c = np.repeat(self.img_3c, 3, axis=2)

        img_1024 = transform.resize(
            self.img_3c,
            (MEDSAM_IMG_INPUT_SIZE, MEDSAM_IMG_INPUT_SIZE),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)

        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)

        # convert the shape to (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            self.embedding = medsam_model.image_encoder(
                img_1024_tensor
            )  # (1, 256, 64, 64)
        print("Done.")


app = QApplication(sys.argv)
w = Window()
w.show()
app.exec()

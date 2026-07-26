import os
import io
import cv2
import torch
import torch.nn as nn
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
import segmentation_models_pytorch as smp

TARGET_SIZE = (512, 512)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

OVERLAY_COLORS = [
    [255, 0, 0],     # MA - أحمر
    [255, 165, 0],   # HE - برتقالي
    [255, 255, 0],   # EX - أصفر
    [255, 0, 255],   # SE - أرجواني
    [0, 255, 0]      # OD - أخضر
]

LESION_NAMES = [
    "Microaneurysms (MA)",
    "Hemorrhages (HE)",
    "Hard Exudates (EX)",
    "Soft Exudates (SE)",
    "Optic Disc (OD)"
]

VESSEL_COLOR = [0, 220, 255]
VESSEL_LABEL = "Blood Vessels"

LEGEND_ITEMS = list(zip(LESION_NAMES, OVERLAY_COLORS)) + [(VESSEL_LABEL, VESSEL_COLOR)]


# --- 1. معالجة الصور والدوال المساعدة ---

def clahe_rgb_safe(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


def preprocess_vessel_image(image_rgb: np.ndarray, target_size=TARGET_SIZE) -> np.ndarray:
    green_channel = image_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green_channel)
    resized_img = cv2.resize(enhanced_green, target_size, interpolation=cv2.INTER_CUBIC)
    return np.stack([resized_img] * 3, axis=-1)


def draw_pred_overlay(img_rgb: np.ndarray, pred_masks: np.ndarray) -> np.ndarray:
    overlay = img_rgb.copy()
    for i, color in enumerate(OVERLAY_COLORS):
        if i < pred_masks.shape[0]:
            overlay[pred_masks[i] > 0] = color
    return overlay


# --- 2. الترميز (Legend) ---

def get_font(size=16):
    """يحاول تحميل خط واضح، ويعود للخط الافتراضي إن لم يجده"""
    candidates = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def add_legend_below(img_rgb: np.ndarray, legend_items=LEGEND_ITEMS) -> np.ndarray:
    """
    يضيف شريط ترميز أسفل الصورة: مربع لوني + اسم كل فئة (آفات + أوعية دموية).
    """
    img_w, img_h = img_rgb.shape[1], img_rgb.shape[0]

    swatch_size = 18
    row_height = 28
    padding_x = 15
    padding_top = 12
    font = get_font(16)

    approx_item_width = 190
    items_per_row = max(1, img_w // approx_item_width)
    n_rows = int(np.ceil(len(legend_items) / items_per_row))

    legend_height = padding_top + n_rows * row_height + padding_top

    canvas = Image.new("RGB", (img_w, img_h + legend_height), (255, 255, 255))
    canvas.paste(Image.fromarray(img_rgb), (0, 0))

    draw = ImageDraw.Draw(canvas)
    item_width = img_w // items_per_row

    for idx, (label, color) in enumerate(legend_items):
        row = idx // items_per_row
        col = idx % items_per_row

        x = padding_x + col * item_width
        y = img_h + padding_top + row * row_height

        draw.rectangle(
            [x, y, x + swatch_size, y + swatch_size],
            fill=tuple(color),
            outline=(0, 0, 0),
        )
        draw.text((x + swatch_size + 8, y + 1), label, fill=(0, 0, 0), font=font)

    return np.array(canvas)


# --- 3. محرك التنبؤ والـ Engine ---

class SegmentationEngine:
    def __init__(self, vessel_model_path: str, lesion_model_path: str):
        # تحميل نموذج الأوعية الدموية (1 قناة للمخرجات)
        self.vessel_model = self._load_model(vessel_model_path, out_channels=1)
        # تحميل نموذج الآفات (5 قنوات للمخرجات)
        self.lesion_model = self._load_model(lesion_model_path, out_channels=5)

    def _load_model(self, model_path: str, out_channels: int):
        """تحميل الملف سواء كان نموذجاً كاملاً أو State Dict (OrderedDict)"""
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Model path does not exist: {model_path}")
            return None
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)

            # إذا كان الملف نموذجاً كاملاً جاهزاً
            if isinstance(checkpoint, nn.Module):
                checkpoint.eval()
                return checkpoint.to(DEVICE)

            # نفس بنية التدريب تمامًا: smp.Unet + efficientnet-b3
            model = smp.Unet(
                encoder_name="efficientnet-b3",
                encoder_weights=None,   # لا حاجة لأوزان imagenet، سيتم استبدالها بالأوزان المدرَّبة
                in_channels=3,
                classes=out_channels,
                activation=None,
            ).to(DEVICE)

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]

            model.load_state_dict(checkpoint)
            model.eval()
            return model

        except Exception as e:
            print(f"❌ Error loading model from {model_path}: {e}")
            return None

    @staticmethod
    def image_to_base64(img_rgb: np.ndarray) -> str:
        pil_img = Image.fromarray(img_rgb)
        buff = io.BytesIO()
        pil_img.save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    @torch.no_grad()
    def predict_lesions(self, orig_rgb: np.ndarray, threshold=0.5):
        # CLAHE أولًا على الدقة الأصلية (كما في التدريب)، ثم resize
        proc_img = clahe_rgb_safe(orig_rgb)
        proc_img = cv2.resize(proc_img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        orig_resized = cv2.resize(orig_rgb, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

        img_tensor = proc_img.astype(np.float32) / 255.0
        img_tensor = (img_tensor - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

        if self.lesion_model is not None:
            output = self.lesion_model(img_tensor)
            pred_masks = (torch.sigmoid(output) > threshold).cpu().numpy()[0].astype(np.uint8)
        else:
            pred_masks = np.zeros((5, TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)

        return orig_resized, pred_masks

    @torch.no_grad()
    def predict_vessels(self, orig_rgb: np.ndarray, threshold=0.5):
        vessel_prep = preprocess_vessel_image(orig_rgb, TARGET_SIZE)
        img_tensor = vessel_prep.astype(np.float32) / 255.0
        img_tensor = (img_tensor - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

        if self.vessel_model is not None:
            output = self.vessel_model(img_tensor)
            vessel_mask = (torch.sigmoid(output) > threshold).cpu().numpy()[0, 0].astype(np.uint8)
        else:
            vessel_mask = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)

        return vessel_mask

    def predict_and_overlay(self, image_bytes: bytes):
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("فشل في قراءة ملف الصورة")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        orig_resized, lesion_masks = self.predict_lesions(img_rgb)
        vessel_mask = self.predict_vessels(img_rgb)

        pred_overlay = draw_pred_overlay(orig_resized, lesion_masks)
        pred_overlay[vessel_mask > 0] = VESSEL_COLOR

        # إضافة شريط الترميز أسفل الصورة
        pred_overlay_with_legend = add_legend_below(pred_overlay)

        total_pixels = TARGET_SIZE[0] * TARGET_SIZE[1]
        vessel_pixels = int(np.sum(vessel_mask > 0))
        vessel_density = round((vessel_pixels / total_pixels) * 100, 2)

        lesions_breakdown = {}
        total_lesion_pixels = 0
        for i, name in enumerate(LESION_NAMES):
            count = int(np.sum(lesion_masks[i] > 0))
            lesions_breakdown[name] = count
            total_lesion_pixels += count

        return {
            "original_image_base64": self.image_to_base64(orig_resized),
            "overlay_image_base64": self.image_to_base64(pred_overlay_with_legend),
            "metrics_summary": {
                "vessel_density_percentage": vessel_density,
                "total_lesion_pixels": total_lesion_pixels,
                "lesions_breakdown": lesions_breakdown
            }
        }

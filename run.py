#!/usr/bin/env python3
"""
run.py — AI-примерка одежды через IDM-VTON

Использование:
    1. Положите фото человека в input/person.jpg
    2. Положите фото одежды в input/cloth.jpg
    3. Запустите: python run.py
    4. Результат: output/result.png
"""

import os
import sys
import shutil
import time
from pathlib import Path

# ─── Пути проекта ───
PROJECT_DIR = Path(__file__).parent.resolve()
IDM_VTON_DIR = PROJECT_DIR / "idm_vton"
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

PERSON_PATH = INPUT_DIR / "person.jpg"
CLOTH_PATH = INPUT_DIR / "cloth.jpg"

# Количество вариантов генерации
NUM_VARIANTS = 3
VARIANT_SEEDS = [42, 123, 456]

# Настройки качества
INFERENCE_STEPS = 40  # 30-50, больше = качественнее но медленнее
GUIDANCE_SCALE = 2.0
WIDTH = 768
HEIGHT = 1024


def validate_inputs():
    """Проверяет наличие и валидность входных файлов."""
    global PERSON_PATH, CLOTH_PATH
    from PIL import Image

    # Проверка файла человека
    if not PERSON_PATH.exists():
        # Проверяем альтернативные расширения
        alt_extensions = ['.jpeg', '.png', '.webp', '.bmp']
        found = False
        for ext in alt_extensions:
            alt_path = INPUT_DIR / f"person{ext}"
            if alt_path.exists():
                found = True
                PERSON_PATH = alt_path
                break
        if not found:
            print("Ошибка: не найден файл input/person.jpg")
            sys.exit(1)

    if not CLOTH_PATH.exists():
        alt_extensions = ['.jpeg', '.png', '.webp', '.bmp']
        found = False
        for ext in alt_extensions:
            alt_path = INPUT_DIR / f"cloth{ext}"
            if alt_path.exists():
                found = True
                CLOTH_PATH = alt_path
                break
        if not found:
            print("Ошибка: не найден файл input/cloth.jpg")
            sys.exit(1)

    # Проверка что файлы являются изображениями
    for fpath, name in [(PERSON_PATH, "person"), (CLOTH_PATH, "cloth")]:
        try:
            img = Image.open(fpath)
            img.verify()
        except Exception:
            print(f"Ошибка: файл {fpath.name} не является изображением")
            sys.exit(1)

    print(f"  ✓ Фото человека: {PERSON_PATH.name} найдено")
    print(f"  ✓ Фото одежды:   {CLOTH_PATH.name} найдено")


def detect_device():
    """Определяет лучшее доступное устройство."""
    import torch

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✓ Устройство: CUDA ({device_name}, {vram:.1f} GB VRAM)")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  ✓ Устройство: MPS (Apple Silicon)")
        print("  ⚠ Внимание: MPS значительно медленнее CUDA. Генерация может занять 30-60 минут.")
        return "mps"
    else:
        print("  ✓ Устройство: CPU")
        print("  ⚠ Внимание: CPU очень медленный. Генерация может занять несколько часов.")
        return "cpu"


def get_torch_dtype(device):
    """Определяет тип данных для устройства."""
    import torch

    if device == "cuda":
        return torch.float16
    elif device == "mps":
        # MPS поддерживает float16 для многих операций, но для стабильности
        # некоторые модели лучше работают в float32
        return torch.float16
    else:
        return torch.float32


def setup_idm_vton_path():
    """Добавляет пути IDM-VTON в sys.path."""
    if not IDM_VTON_DIR.exists():
        print("Ошибка: директория idm_vton не найдена.")
        print("Запустите сначала: ./setup.sh")
        sys.exit(1)

    # Добавляем пути для импорта модулей IDM-VTON
    idm_path = str(IDM_VTON_DIR)
    if idm_path not in sys.path:
        sys.path.insert(0, idm_path)

    # Также добавляем gradio_demo для доступа к utils_mask
    gradio_demo_path = str(IDM_VTON_DIR / "gradio_demo")
    if gradio_demo_path not in sys.path:
        sys.path.insert(0, gradio_demo_path)


def load_preprocessing_models(device):
    """Загружает модели препроцессинга (OpenPose, Human Parsing)."""
    print("  Загрузка OpenPose...")
    from preprocess.openpose.run_openpose import OpenPose

    # OpenPose использует индекс GPU (0) или -1 для CPU
    gpu_id = 0 if device == "cuda" else -1
    openpose_model = OpenPose(gpu_id)

    print("  Загрузка Human Parsing...")
    from preprocess.humanparsing.run_parsing import Parsing

    parsing_model = Parsing(gpu_id)

    print("  ✓ Модели препроцессинга загружены")
    return openpose_model, parsing_model


def preprocess_person(human_img, openpose_model, parsing_model, device):
    """
    Выполняет препроцессинг фото человека:
    - OpenPose для определения позы
    - Human Parsing для сегментации
    - DensePose для 3D-поверхности тела
    - Вычисление маски одежды
    """
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image

    from utils_mask import get_mask_location

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # ─── OpenPose + Human Parsing (на уменьшенном изображении) ───
    print("  Определение позы (OpenPose)...")
    small_img = human_img.resize((384, 512))
    keypoints = openpose_model(small_img)

    print("  Сегментация тела (Human Parsing)...")
    model_parse, _ = parsing_model(small_img)

    # ─── Маска одежды ───
    print("  Вычисление маски одежды...")
    mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
    mask = mask.resize((WIDTH, HEIGHT))

    # Вычисляем mask_gray для визуализации
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    # ─── DensePose ───
    print("  Вычисление DensePose...")
    pose_img = compute_densepose(human_img, device)
    pose_img = pose_img.resize((WIDTH, HEIGHT))

    return mask, mask_gray, pose_img, tensor_transform


def compute_densepose(human_img, device):
    """Вычисляет DensePose изображение."""
    import numpy as np
    from PIL import Image

    try:
        from detectron2.data.detection_utils import (
            _apply_exif_orientation,
            convert_PIL_to_numpy,
        )
        import apply_net

        # DensePose всегда на CPU для macOS совместимости
        densepose_device = "cuda" if device == "cuda" else "cpu"

        human_img_resized = human_img.resize((384, 512))
        human_img_arg = _apply_exif_orientation(human_img_resized)
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

        # Путь к конфигурации и чекпоинту DensePose
        config_path = str(IDM_VTON_DIR / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml")
        model_path = str(IDM_VTON_DIR / "ckpt" / "densepose" / "model_final_162be9.pkl")

        args = apply_net.create_argument_parser().parse_args((
            'show', config_path, model_path, 'dp_segm',
            '-v', '--opts', 'MODEL.DEVICE', densepose_device
        ))

        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]  # BGR -> RGB
        pose_img = Image.fromarray(pose_img)

        return pose_img

    except Exception as e:
        print(f"  ⚠ DensePose не удалось вычислить: {e}")
        print("  Используем пустое pose-изображение (качество может снизиться)...")
        # Возвращаем чёрное изображение как fallback
        return Image.new("RGB", (384, 512), (0, 0, 0))


def load_pipeline(device, dtype):
    """Загружает основную модель IDM-VTON."""
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import (
        AutoTokenizer,
        CLIPImageProcessor,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    )

    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel

    base_path = "yisol/IDM-VTON"

    print("  Загрузка UNet (TryonNet)...")
    unet = UNet2DConditionModel.from_pretrained(
        base_path, subfolder="unet", torch_dtype=dtype,
    )
    unet.requires_grad_(False)

    print("  Загрузка токенизаторов...")
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path, subfolder="tokenizer", revision=None, use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path, subfolder="tokenizer_2", revision=None, use_fast=False,
    )

    print("  Загрузка планировщика шума...")
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    print("  Загрузка текстовых энкодеров (CLIP)...")
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path, subfolder="text_encoder", torch_dtype=dtype,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path, subfolder="text_encoder_2", torch_dtype=dtype,
    )

    print("  Загрузка энкодера изображений (CLIP Vision)...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path, subfolder="image_encoder", torch_dtype=dtype,
    )

    print("  Загрузка VAE...")
    vae = AutoencoderKL.from_pretrained(
        base_path, subfolder="vae", torch_dtype=dtype,
    )

    print("  Загрузка UNet Encoder (GarmentNet)...")
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path, subfolder="unet_encoder", torch_dtype=dtype,
    )

    # Замораживаем все модели
    unet_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    print("  Сборка пайплайна...")
    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    pipe.unet_encoder = unet_encoder

    # Перемещаем на устройство с оптимизацией памяти для CUDA
    if device == "cuda":
        import torch

        print("  Включение оптимизации VRAM (CPU Offload & VAE Slicing)...")

        # ── Патч 1: encoder_hid_proj ──
        # В tryon_pipeline.py (строка ~1726) encoder_hid_proj вызывается
        # как self.unet.encoder_hid_proj(image_embeds) — напрямую, без
        # вызова self.unet(), поэтому хук CPU Offload не срабатывает и
        # encoder_hid_proj остаётся на CPU. Патчим его forward, чтобы
        # он сам переносился на GPU перед выполнением.
        _original_ehp_forward = pipe.unet.encoder_hid_proj.forward
        def _ehp_forward_wrapper(*args, **kwargs):
            pipe.unet.encoder_hid_proj.to(device)
            return _original_ehp_forward(*args, **kwargs)
        pipe.unet.encoder_hid_proj.forward = _ehp_forward_wrapper

        # Включаем CPU Offload с правильной цепочкой моделей
        pipe.model_cpu_offload_seq = (
            "text_encoder->text_encoder_2->image_encoder"
            "->unet_encoder->unet->vae"
        )
        pipe.enable_model_cpu_offload()

        # ── Патч 2: чередование UNet-ов в цикле денойзинга ──
        # В цикле на каждом шаге вызываются unet_encoder, затем unet.
        # Хук CPU Offload — линейная цепочка: pre_forward каждой модели
        # выгружает только ПРЕДЫДУЩУЮ модель в цепочке.
        # Для unet_encoder предыдущая — image_encoder, а НЕ unet.
        # Поэтому на 2-й итерации unet остаётся на GPU (~5 ГБ) и
        # при загрузке unet_encoder (~5 ГБ) происходит OOM.
        # Фикс: патчим pre_forward хука unet_encoder, чтобы он
        # дополнительно выгружал unet перед своей загрузкой.
        _orig_ue_pre_forward = pipe.unet_encoder._hf_hook.pre_forward

        def _patched_ue_pre_forward(module, *args, **kwargs):
            pipe.unet.to("cpu")  # unet → CPU
            torch.cuda.empty_cache()
            return _orig_ue_pre_forward(module, *args, **kwargs)

        pipe.unet_encoder._hf_hook.pre_forward = _patched_ue_pre_forward

        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    else:
        print(f"  Перемещение моделей на {device}...")
        pipe.to(device)
        pipe.unet_encoder.to(device)

    print("  ✓ Пайплайн загружен")
    return pipe


def run_inference(pipe, human_img, garm_img, mask, pose_img, tensor_transform,
                  device, dtype, seed, step_num, total_steps):
    """Выполняет одну генерацию с заданным seed."""
    import torch
    from typing import List

    print(f"  [{step_num}/{total_steps}] Генерация (seed={seed}, steps={INFERENCE_STEPS})...")

    garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, dtype)
    pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, dtype)

    from contextlib import nullcontext as _nullcontext

    autocast_ctx = torch.cuda.amp.autocast() if device == "cuda" else _nullcontext()

    with torch.no_grad(), autocast_ctx:
        # Промпт для TryonNet (описание человека в одежде)
        prompt = "model is wearing upper body garment"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        # Промпт для GarmentNet (описание одежды)
        cloth_prompt = "a photo of upper body garment"
        if not isinstance(cloth_prompt, List):
            cloth_prompt = [cloth_prompt] * 1
        neg_prompt_list = [negative_prompt] * 1

        prompt_embeds_c, _, _, _ = pipe.encode_prompt(
            cloth_prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=neg_prompt_list,
        )

        if device == "cuda":
            generator = torch.Generator(device).manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)

        images = pipe(
            prompt_embeds=prompt_embeds.to(device, dtype),
            negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
            pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
            num_inference_steps=INFERENCE_STEPS,
            generator=generator,
            strength=1.0,
            pose_img=pose_tensor,
            text_embeds_cloth=prompt_embeds_c.to(device, dtype),
            cloth=garm_tensor,
            mask_image=mask,
            image=human_img,
            height=HEIGHT,
            width=WIDTH,
            ip_adapter_image=garm_img.resize((WIDTH, HEIGHT)),
            guidance_scale=GUIDANCE_SCALE,
        )[0]

    return images[0]




def select_best_result(variants, garm_img):
    """
    Выбирает лучший результат из нескольких вариантов.
    Использует CLIP для сравнения результата с одеждой.
    """
    if len(variants) == 1:
        return 0

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        print("  Оценка вариантов через CLIP...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # Вычисляем сходство каждого варианта с одеждой
        scores = []
        for i, variant in enumerate(variants):
            inputs = processor(
                images=[variant, garm_img],
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                image_features = model.get_image_features(**{
                    k: v for k, v in inputs.items()
                    if k == "pixel_values"
                })
                # Нормализуем
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # Сходство между результатом и одеждой
                similarity = (image_features[0] @ image_features[1]).item()
                scores.append(similarity)
                print(f"    Вариант {i + 1}: CLIP score = {similarity:.4f}")

        best_idx = scores.index(max(scores))
        print(f"  ✓ Лучший вариант: {best_idx + 1}")
        return best_idx

    except Exception as e:
        print(f"  ⚠ CLIP оценка не удалась: {e}")
        print("  Используем первый вариант как лучший")
        return 0


def prepare_images():
    """Загружает и подготавливает входные изображения."""
    from PIL import Image

    print("  Загрузка фото человека...")
    human_img_orig = Image.open(PERSON_PATH).convert("RGB")

    # Smart crop: обрезаем до соотношения 3:4 если нужно
    width, height = human_img_orig.size
    target_width = int(min(width, height * (3 / 4)))
    target_height = int(min(height, width * (4 / 3)))
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2

    cropped_img = human_img_orig.crop((left, top, right, bottom))
    crop_size = cropped_img.size
    human_img = cropped_img.resize((WIDTH, HEIGHT))

    print(f"    Оригинал: {width}x{height} → Обрезка: {target_width}x{target_height} → Ресайз: {WIDTH}x{HEIGHT}")

    print("  Загрузка фото одежды...")
    garm_img = Image.open(CLOTH_PATH).convert("RGB").resize((WIDTH, HEIGHT))

    return human_img_orig, human_img, garm_img, (left, top, right, bottom), crop_size


def paste_result_on_original(result_img, human_img_orig, crop_coords, crop_size):
    """Вставляет результат обратно на оригинальное фото (если было обрезано)."""
    left, top, right, bottom = crop_coords
    out_img = result_img.resize(crop_size)
    output = human_img_orig.copy()
    output.paste(out_img, (int(left), int(top)))
    return output


def main():
    """Главная функция."""
    start_time = time.time()

    print("")
    print("╔══════════════════════════════════════════╗")
    print("║   IDM-VTON — AI-примерка одежды          ║")
    print("╚══════════════════════════════════════════╝")
    print("")

    # ─── 1. Валидация входных данных ───
    print("▸ Проверка входных файлов...")
    validate_inputs()

    # ─── 2. Определение устройства ───
    print("")
    print("▸ Определение устройства...")
    device = detect_device()

    import torch
    dtype = get_torch_dtype(device)

    # ─── 3. Настройка путей IDM-VTON ───
    print("")
    print("▸ Настройка IDM-VTON...")
    setup_idm_vton_path()

    # Переходим в директорию IDM-VTON для корректных путей
    original_cwd = os.getcwd()
    os.chdir(IDM_VTON_DIR)

    try:
        # ─── 4. Подготовка изображений ───
        print("")
        print("▸ Подготовка изображений...")
        human_img_orig, human_img, garm_img, crop_coords, crop_size = prepare_images()

        # ─── 5. Загрузка моделей препроцессинга ───
        print("")
        print("▸ Загрузка моделей препроцессинга...")
        openpose_model, parsing_model = load_preprocessing_models(device)

        # ─── 6. Препроцессинг ───
        print("")
        print("▸ Препроцессинг фото человека...")
        mask, mask_gray, pose_img, tensor_transform = preprocess_person(
            human_img, openpose_model, parsing_model, device
        )

        # ─── 7. Загрузка основного пайплайна ───
        print("")
        print("▸ Загрузка модели IDM-VTON (при первом запуске скачивает ~15 ГБ)...")
        pipe = load_pipeline(device, dtype)

        # Перемещаем OpenPose на устройство если CUDA
        if device == "cuda":
            openpose_model.preprocessor.body_estimation.model.to(device)

        # ─── 8. Генерация вариантов ───
        print("")
        print(f"▸ Генерация {NUM_VARIANTS} вариантов...")
        variants = []

        for i, seed in enumerate(VARIANT_SEEDS[:NUM_VARIANTS]):
            result = run_inference(
                pipe, human_img, garm_img, mask, pose_img, tensor_transform,
                device, dtype, seed, i + 1, NUM_VARIANTS
            )

            # Вставляем результат на оригинальное фото
            final = paste_result_on_original(result, human_img_orig, crop_coords, crop_size)
            variants.append(final)

            # Сохраняем вариант
            variant_path = OUTPUT_DIR / f"result_{i + 1}.png"
            final.save(variant_path, quality=95)
            print(f"    ✓ Сохранён: {variant_path}")

            # Полная очистка GPU между генерациями вариантов.
            # После pipe() на GPU остаются все модели, загруженные хуками
            # (VAE, text_encoder и др.). Нужно вернуть всё на CPU.
            if device == "cuda":
                import gc
                import torch
                for name in ["unet", "unet_encoder", "vae",
                             "text_encoder", "text_encoder_2", "image_encoder"]:
                    m = getattr(pipe, name, None)
                    if m is not None:
                        m.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()

        # ─── 9. Выбор лучшего результата ───
        print("")
        print("▸ Выбор лучшего результата...")
        best_idx = select_best_result(variants, garm_img)
        best_result = variants[best_idx]

        # Сохраняем лучший результат
        result_path = OUTPUT_DIR / "result.png"
        best_result.save(result_path, quality=95)
        print(f"  ✓ Лучший результат сохранён: {result_path}")

        # ─── Итого ───
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print("")
        print("╔══════════════════════════════════════════╗")
        print("║   Генерация завершена!                    ║")
        print("╚══════════════════════════════════════════╝")
        print("")
        print(f"  Время: {minutes} мин {seconds} сек")
        print(f"  Варианты: output/result_1.png ... result_{NUM_VARIANTS}.png")
        print(f"  Лучший:   output/result.png (вариант {best_idx + 1})")
        print("")

    except Exception as e:
        print("")
        print(f"Ошибка генерации. Попробуйте заменить входные фото на более качественные.")
        print(f"Детали: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()

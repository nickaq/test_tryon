#!/usr/bin/env bash
#
# setup.sh — Одноразовая настройка проекта IDM-VTON
#
# Использование:
#   chmod +x setup.sh
#   ./setup.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
    echo "Активируем виртуальное окружение .venv..."
    source .venv/bin/activate
fi

# Helper function to check if file exists and is larger than 100KB (not a git placeholder)
is_valid_file() {
    local file="$1"
    if [ -f "$file" ]; then
        local size
        size=$(wc -c < "$file" | tr -d '[:space:]')
        if [ "$size" -gt 100000 ]; then
            return 0
        fi
    fi
    return 1
}

# Helper function to download file using curl or wget
download_file() {
    local url="$1"
    local output="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -L --progress-bar -o "$output" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$output" "$url"
    else
        echo "Ошибка: не найден ни curl, ни wget. Невозможно скачать файлы." >&2
        return 1
    fi
}


echo "========================================"
echo "  Настройка IDM-VTON Virtual Try-On"
echo "========================================"
echo ""

# ─── 1. Клонирование репозитория IDM-VTON ───
if [ ! -d "idm_vton" ]; then
    echo "[1/5] Клонирование репозитория IDM-VTON..."
    git clone https://github.com/yisol/IDM-VTON.git idm_vton
    echo "  ✓ Репозиторий клонирован"
else
    echo "[1/5] Репозиторий IDM-VTON уже существует, пропускаем..."
fi

# ─── 2. Установка pip-зависимостей (включая PyTorch) ───
echo ""
echo "[2/5] Установка Python-зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt
echo "  ✓ Зависимости установлены"

# ─── 3. Установка detectron2 (требует уже установленный torch) ───
echo ""
echo "[3/5] Установка detectron2..."
if python -c "import detectron2" 2>/dev/null; then
    echo "  ✓ detectron2 уже установлен"
else
    echo "  Сборка detectron2 из исходного кода (torch уже установлен)..."
    # Определяем ОС для правильных флагов компиляции
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "  Обнаружена macOS, используем clang..."
        CC=clang CXX=clang++ pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
    else
        pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
    fi
    echo "  ✓ detectron2 установлен"
fi

# ─── 4. Скачивание чекпоинтов для препроцессинга ───
echo ""
echo "[4/5] Скачивание чекпоинтов..."

CKPT_DIR="idm_vton/ckpt"

# DensePose checkpoint
DENSEPOSE_DIR="$CKPT_DIR/densepose"
DENSEPOSE_FILE="$DENSEPOSE_DIR/model_final_162be9.pkl"
if ! is_valid_file "$DENSEPOSE_FILE"; then
    echo "  Скачивание DensePose модели..."
    mkdir -p "$DENSEPOSE_DIR"
    download_file "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl" "$DENSEPOSE_FILE"
    echo "  ✓ DensePose модель скачана"
else
    echo "  ✓ DensePose модель уже существует"
fi

# Human Parsing checkpoints
PARSING_DIR="$CKPT_DIR/humanparsing"
if ! is_valid_file "$PARSING_DIR/parsing_atr.onnx" || ! is_valid_file "$PARSING_DIR/parsing_lip.onnx"; then
    echo "  Скачивание моделей Human Parsing..."
    mkdir -p "$PARSING_DIR"
    download_file "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx" "$PARSING_DIR/parsing_atr.onnx"
    download_file "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx" "$PARSING_DIR/parsing_lip.onnx"
    echo "  ✓ Human Parsing модели скачаны"
else
    echo "  ✓ Human Parsing модели уже существуют"
fi

# OpenPose checkpoint
OPENPOSE_DIR="$CKPT_DIR/openpose/ckpts"
if ! is_valid_file "$OPENPOSE_DIR/body_pose_model.pth"; then
    echo "  Скачивание OpenPose модели..."
    mkdir -p "$OPENPOSE_DIR"
    download_file "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth" "$OPENPOSE_DIR/body_pose_model.pth"
    echo "  ✓ OpenPose модель скачана"
else
    echo "  ✓ OpenPose модель уже существует"
fi

# DensePose config
DENSEPOSE_CFG="idm_vton/configs"
if [ ! -f "$DENSEPOSE_CFG/densepose_rcnn_R_50_FPN_s1x.yaml" ]; then
    echo "  Скачивание конфигурации DensePose..."
    mkdir -p "$DENSEPOSE_CFG"
    download_file "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml" "$DENSEPOSE_CFG/densepose_rcnn_R_50_FPN_s1x.yaml"
    echo "  ✓ Конфигурация DensePose скачана"
else
    echo "  ✓ Конфигурация DensePose уже существует"
fi

# ─── 5. Создание директорий ───
echo ""
echo "[5/5] Создание директорий..."
mkdir -p input output
echo "  ✓ Директории созданы"

echo ""
echo "========================================"
echo "  Настройка завершена!"
echo "========================================"
echo ""
echo "Следующие шаги:"
echo "  1. Положите фото человека: input/person.jpg"
echo "  2. Положите фото одежды:  input/cloth.jpg"
echo "  3. Запустите: python run.py"
echo ""
echo "ВНИМАНИЕ: При первом запуске run.py будут скачаны"
echo "модели с HuggingFace (~15 ГБ). Это может занять время."
echo ""

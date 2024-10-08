#!/bin/bash
# script to create a model artifact for SageMaker inference

set -e # Exit on error
set -u # Exit on undefined variable
# set -x # Print commands

# handy function to download files from hugging face
# usage: download_huggingface <url> <target folder>
download_huggingface() {
    # first wget with --no-clobber, then wget with --timestamping
    wget -nc "$1" -P "$2" || wget -N "$1" -P "$2"
    # wget --header="Authorization: Bearer ${HF_TOKEN}" -nc "$1" -P "$2" || wget --header="Authorization: Bearer ${HF_TOKEN}" -N "$1" -P "$2"
}

# Function to download files from any URL
# usage: download_file <url> <output_path>
download_file() {
    wget -nc "$1" -O "$2" || wget -N "$1" -O "$2"
}

function checkout_gitrepo() {
  echo "########################################"
  echo "Checking out $1... at rivision $2 to $3"
  echo "Base directory: $4"

  cd $4
  git clone --no-tags --recurse-submodules --shallow-submodules $1
  cd $3
  git reset --hard $2

  echo "########################################"
}

# target folder for downloading model artifact
TARGET_DIR="model-artifact"

# target file for tar-gzip archive of model artifact
TARGET_FILE="model-artifact.tgz"

show_usage() {
    echo "Usage: $0 [s3://path/to/s3/object]"
    exit 1
}
# s3 upload path (optional)
S3_PATH=""
if [ "$#" -gt 1 ]; then
    show_usage
elif [ "$#" -eq 1 ]; then
    if [[ "$1" == s3://* ]]; then
        S3_PATH="$1"
    else
        show_usage
    fi
fi

# initialize empty folder structure
mkdir -p "${TARGET_DIR}"
DIRS=(
    checkpoints clip clip_vision configs controlnet embeddings loras upscale_models vae gligen unet opencv_3rdparty input
    vae checkpoints embeddings controlnet loras opencv_3rdparty upscale_models clip_vision
    insightface facerestore_models ultralytics/bbox ella
    t5_model/flan-t5-xl-sharded-bf16 custom_nodes
)
for dir in "${DIRS[@]}"
do
    mkdir -p "${TARGET_DIR}/${dir}"
done

# download models that you want to include
# stable-diffusion-xl-base-1.0 
# download_huggingface 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors' "${TARGET_DIR}/checkpoints"

# Flux Dev (fp8 checkpoint version)
# Ref: https://comfyanonymous.github.io/ComfyUI_examples/flux/#flux-dev-1
# download_huggingface 'https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors' "${TARGET_DIR}/checkpoints"

# Flux Schnell (fp8 checkpoint version)
# Ref: https://comfyanonymous.github.io/ComfyUI_examples/flux/#flux-schnell-1
# download_huggingface 'https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors' "${TARGET_DIR}/checkpoints"

# black-forest-labs/FLUX.1-dev (requires authentication)
# download_huggingface 'https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors' "${TARGET_DIR}/unet" 
# download_huggingface 'https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors' "${TARGET_DIR}/vae"
# download_huggingface 'https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors' "${TARGET_DIR}/clip"
# download_huggingface 'https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors' "${TARGET_DIR}/clip"

# black-forest-labs/FLUX.1-schnell (requires authentication)
# download_huggingface 'https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors' "${TARGET_DIR}/unet" 
# download_huggingface 'https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors' "${TARGET_DIR}/vae"
# download_huggingface 'https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors' "${TARGET_DIR}/clip"
# download_huggingface 'https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors' "${TARGET_DIR}/clip"

# ComfyUI-Manager - extension to manage custom nodes
# cd ${TARGET_DIR}/custom_nodes
# [[ -e ComfyUI-Manager ]] || git clone https://github.com/ltdrdata/ComfyUI-Manager.git && (cd ComfyUI-Manager && git fetch && git checkout 2.48.6)
#



# New downloads
download_file 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors' "${TARGET_DIR}/vae/vae-ft-mse-840000-ema-pruned.safetensors"
download_file 'https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.ckpt' "${TARGET_DIR}/vae/vae-ft-ema-560000-ema-pruned.ckpt"

download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/checkpoints/Deliberate_v2.safetensors' "${TARGET_DIR}/checkpoints/Deliberate_v2.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/checkpoints/Realistic_Vision_V5.1_fp16-no-ema.safetensors' "${TARGET_DIR}/checkpoints/Realistic_Vision_V5.1_fp16-no-ema.safetensors"
download_file 'https://huggingface.co/swl-models/Anything-v5.0-PRT/resolve/main/Anything-v5.0-PRT-RE.safetensors' "${TARGET_DIR}/checkpoints/Anything-v5.0-PRT-RE.safetensors"
download_file 'https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors' "${TARGET_DIR}/checkpoints/DreamShaper_8_pruned.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/checkpoints/analogDiffusion_10Safetensors.safetensors' "${TARGET_DIR}/checkpoints/analogDiffusion_10Safetensors.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/checkpoints/analogMadness_v60.safetensors' "${TARGET_DIR}/checkpoints/analogMadness_v60.safetensors"
download_file 'https://civitai.com/api/download/models/274039' "${TARGET_DIR}/checkpoints/juggernaut_reborn.safetensors"

download_file 'https://photoshot-us.s3.amazonaws.com/embeding/BadDream.pt' "${TARGET_DIR}/embeddings/BadDream.pt"
download_file 'https://photoshot-us.s3.amazonaws.com/embeding/bad-hands-5.pt' "${TARGET_DIR}/embeddings/bad-hands-5.pt"
download_file 'https://photoshot-us.s3.amazonaws.com/embeding/bad_pictures.pt' "${TARGET_DIR}/embeddings/bad_pictures.pt"
download_file 'https://photoshot-us.s3.amazonaws.com/embeding/badhandv4.pt' "${TARGET_DIR}/embeddings/badhandv4.pt"
download_file 'https://photoshot-us.s3.amazonaws.com/embeding/easynegative.safetensors' "${TARGET_DIR}/embeddings/easynegative.safetensors"
download_file 'https://photoshot-us.s3.amazonaws.com/embeding/negative_hand-neg.pt' "${TARGET_DIR}/embeddings/negative_hand-neg.pt"
download_file 'https://photoshot-us.s3.amazonaws.com/embeding/ng_deepnegative_v1_75t.pt' "${TARGET_DIR}/embeddings/ng_deepnegative_v1_75t.pt"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/embeddings/16-token-negative-deliberate-neg.pt' "${TARGET_DIR}/embeddings/16-token-negative-deliberate-neg.pt"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/embeddings/JuggernautNegative-neg.pt' "${TARGET_DIR}/embeddings/JuggernautNegative-neg.pt"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/embeddings/verybadimagenegative_v1.3.pt' "${TARGET_DIR}/embeddings/verybadimagenegative_v1.3.pt"

download_file 'https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.safetensors?download=true' "${TARGET_DIR}/controlnet/control_v11p_sd15_openpose.safetensors"
download_file 'https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/v2/control_v1p_sd15_qrcode_monster_v2.safetensors?download=true' "${TARGET_DIR}/controlnet/control_v1p_sd15_qrcode_monster_v2.safetensors"
download_file 'https://huggingface.co/ioclab/ioc-controlnet/resolve/main/models/control_v1p_sd15_brightness.safetensors' "${TARGET_DIR}/controlnet/control_v1p_sd15_brightness.safetensors"
download_file 'https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth' "${TARGET_DIR}/controlnet/control_v11p_sd15_canny.pth"

download_file 'https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth' "${TARGET_DIR}/controlnet/control_v11f1e_sd15_tile.pth"
download_file 'https://huggingface.co/DionTimmer/controlnet_qrcode/resolve/main/control_v1p_sd15_qrcode.safetensors' "${TARGET_DIR}/controlnet/control_v1p_sd15_qrcode.safetensors"

download_file 'https://huggingface.co/Nacholmo/controlnet-qr-pattern/resolve/main/Automatic1111-Compatible/control_v20e_sd15_qr_pattern.safetensors' "${TARGET_DIR}/controlnet/control_v20e_sd15_qr_pattern.safetensors"
download_file 'https://huggingface.co/Nacholmo/controlnet-qr-pattern/resolve/main/Automatic1111-Compatible/control_v20e_sd15_qr_pattern.yaml' "${TARGET_DIR}/controlnet/control_v20e_sd15_qr_pattern.yaml"
download_file 'https://huggingface.co/DionTimmer/controlnet_qrcode/resolve/main/control_v1p_sd15_qrcode.yaml' "${TARGET_DIR}/controlnet/control_v1p_sd15_qrcode.yaml"
download_file 'https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.yaml' "${TARGET_DIR}/controlnet/control_v11f1e_sd15_tile.yaml"
download_file 'https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth' "${TARGET_DIR}/controlnet/control_v11f1p_sd15_depth.pth"
download_file 'https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.yaml' "${TARGET_DIR}/controlnet/control_v11f1p_sd15_depth.yaml"
download_file 'https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.yaml' "${TARGET_DIR}/controlnet/control_v11p_sd15_canny.yaml"
download_file 'https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.yaml' "${TARGET_DIR}/controlnet/control_v11p_sd15_openpose.yaml"
download_file 'https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/coadapter-color-sd15v1.pth' "${TARGET_DIR}/controlnet/coadapter-color-sd15v1.pth"
download_file 'https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth' "${TARGET_DIR}/controlnet/control_v11p_sd15_openpose.pth"

download_file 'https://civitai.com/api/download/models/87153' "${TARGET_DIR}/loras/more_details.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/lit.safetensors' "${TARGET_DIR}/loras/lit.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/school_yearbook_photos.safetensors' "${TARGET_DIR}/loras/school_yearbook_photos.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/BetterGuns-V1.safetensors' "${TARGET_DIR}/loras/BetterGuns-V1.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/StealthTech-20.safetensors' "${TARGET_DIR}/loras/StealthTech-20.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/VampiricTech-20.safetensors' "${TARGET_DIR}/loras/VampiricTech-20.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/GlyphTech-20.safetensors' "${TARGET_DIR}/loras/GlyphTech-20.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/BlessedTech-20.safetensors' "${TARGET_DIR}/loras/BlessedTech-20.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/KawaiiTech-20.safetensors' "${TARGET_DIR}/loras/KawaiiTech-20.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/GoldenTech-20.safetensors' "${TARGET_DIR}/loras/GoldenTech-20.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/HackedTech-20.safetensors' "${TARGET_DIR}/loras/HackedTech-20.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/DreadTech-20.safetensors' "${TARGET_DIR}/loras/DreadTech-20.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/PlushTech-22.safetensors' "${TARGET_DIR}/loras/PlushTech-22.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/Mecha.safetensors' "${TARGET_DIR}/loras/Mecha.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/animeoutlineV4_16.safetensors' "${TARGET_DIR}/loras/animeoutlineV4_16.safetensors"
download_file 'https://purrfect-ai-use.s3.amazonaws.com/ai-models/loras/GachaSplash4.safetensors' "${TARGET_DIR}/loras/GachaSplash4.safetensors"

download_file 'https://huggingface.co/Nacholmo/controlnet-qr-pattern-v2/resolve/main/automatic1111/QRPattern_v2_9500.safetensors' "${TARGET_DIR}/controlnet/QRPattern_v2_9500.safetensors"
download_file 'https://huggingface.co/Nacholmo/controlnet-qr-pattern-v2/resolve/main/automatic1111/QRPattern_v2_9500.yaml' "${TARGET_DIR}/controlnet/QRPattern_v2_9500.yaml"
download_file 'https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/control_v1p_sd15_qrcode_monster.safetensors' "${TARGET_DIR}/controlnet/control_v1p_sd15_qrcode_monster.safetensors"

download_file 'https://photoshot-us.s3.amazonaws.com/comfy/opencv_3rdparty/detect.caffemodel' "${TARGET_DIR}/opencv_3rdparty/detect.caffemodel"
download_file 'https://photoshot-us.s3.amazonaws.com/comfy/opencv_3rdparty/detect.prototxt' "${TARGET_DIR}/opencv_3rdparty/detect.prototxt"
download_file 'https://photoshot-us.s3.amazonaws.com/comfy/opencv_3rdparty/sr.caffemodel' "${TARGET_DIR}/opencv_3rdparty/sr.caffemodel"
download_file 'https://photoshot-us.s3.amazonaws.com/comfy/opencv_3rdparty/sr.prototxt' "${TARGET_DIR}/opencv_3rdparty/sr.prototxt"

download_file 'https://huggingface.co/datasets/Kizi-Art/Upscale/resolve/fa98e357882a23b8e7928957a39462fbfaee1af5/4x-UltraSharp.pth' "${TARGET_DIR}/upscale_models/4x-UltraSharp.pth"

download_file 'https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors' "${TARGET_DIR}/clip_vision/ip_adapter_vision.safetensors"


download_file 'https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors' "${TARGET_DIR}/loras/ip-adapter-faceid-plusv2_sd15_lora.safetensors"

download_file 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx' "${TARGET_DIR}/insightface/inswapper_128.onnx"

download_file 'https://civitai.com/api/download/models/239983?type=Model&format=SafeTensor&size=pruned&fp=fp16' "${TARGET_DIR}/checkpoints/Swizz8-XART-BakedVAE-FP16-Pruned.safetensors"

download_file 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth' "${TARGET_DIR}/facerestore_models/GFPGANv1.4.pth"

download_file 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt' "${TARGET_DIR}/ultralytics/bbox/face_yolov8m.pt"
download_file 'https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt' "${TARGET_DIR}/ultralytics/bbox/hand_yolov8s.pt"
download_file 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt' "${TARGET_DIR}/ultralytics/bbox/person_yolov8m-seg.pt"

download_file 'https://huggingface.co/QQGYLab/ELLA/resolve/main/ella-sd1.5-tsc-t5xl.safetensors?download=true' "${TARGET_DIR}/ella/ella-sd1.5-tsc-t5xl.safetensors"
download_file 'https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16/resolve/main/.gitattributes?download=true' "${TARGET_DIR}/t5_model/flan-t5-xl-sharded-bf16/.gitattributes"
download_file 'https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16/resolve/main/config.json?download=true' "${TARGET_DIR}/t5_model/flan-t5-xl-sharded-bf16/config.json"
download_file 'https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16/resolve/main/generation_config.json?download=true' "${TARGET_DIR}/t5_model/flan-t5-xl-sharded-bf16/generation_config.json"
download_file 'https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16/resolve/main/pytorch_model-00001-of-00003.bin?download=true' "${TARGET_DIR}/t5_model/flan-t5-xl-sharded-bf16/pytorch_model-00001-of-00003.bin"
download_file 'https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16/resolve/main/pytorch_model-00002-of-00003.bin?download=true' "${TARGET_DIR}/t5_model/flan-t5-xl-sharded-bf16/pytorch_model-00002-of-00003.bin"
download_file 'https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16/resolve/main/pytorch_model-00003-of-00003.bin?download=true' "${TARGET_DIR}/t5_model/flan-t5-xl-sharded-bf16/pytorch_model-00003-of-00003.bin"
download_file 'https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16/resolve/main/pytorch_model.bin.index.json?download=true' "${TARGET_DIR}/t5_model/flan-t5-xl-sharded-bf16/pytorch_model.bin.index.json"
download_file 'https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16/resolve/main/special_tokens_map.json?download=true' "${TARGET_DIR}/t5_model/flan-t5-xl-sharded-bf16/special_tokens_map.json"
download_file 'https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16/resolve/main/spiece.model?download=true' "${TARGET_DIR}/t5_model/flan-t5-xl-sharded-bf16/spiece.model"
download_file 'https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16/resolve/main/tokenizer.json?download=true' "${TARGET_DIR}/t5_model/flan-t5-xl-sharded-bf16/tokenizer.json"
download_file 'https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16/resolve/main/tokenizer_config.json?download=true' "${TARGET_DIR}/t5_model/flan-t5-xl-sharded-bf16/tokenizer_config.json"wnload_file 'https://civitai.com/api/download/models/274039' "${TARGET_DIR}/checkpoints/juggernaut_reborn.safetensors"


checkout_gitrepo https://github.com/Fannovel16/comfyui_controlnet_aux.git "14c4feda42e218cf026d699d8d6c65917d73aa61" comfyui_controlnet_aux "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/cubiq/ComfyUI_IPAdapter_plus.git "4b6d3c3518258e543af75a1731084f2db3157f70" ComfyUI_IPAdapter_plus "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/lienminhquang/was-node-suite-comfyui.git "e7a74db7916b183981c358911557d5f55893b849" was-node-suite-comfyui "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/Nourepide/ComfyUI-Allor.git "b7fb9ff0bd50124afcbf8cd2638a73f883d32a23" ComfyUI-Allor "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/lienminhquang/ComfyUI-Impact-Pack.git "f6a5d0c02f7c9a6dec4ef95220112e85b73bbb00" ComfyUI-Impact-Pack "${TARGET_DIR}/custom_nodes"
# checkout_gitrepo https://github.com/PurrfectAI/comfyUI-custom-nodes.git "5cb3c6dcff47017b006f8f09312a0c1cbeaef15f" comfyUI-custom-nodes "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/Gourieff/comfyui-reactor-node.git "161585495271c14fdf4b299f93e4c4fe5be1d2cc" comfyui-reactor-node "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/jags111/efficiency-nodes-comfyui.git "a6df20f4c60af49f08bbe122bc486c286f5d0745" efficiency-nodes-comfyui "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/ceruleandeep/ComfyUI-LLaVA-Captioner.git "8fcd56888ca9eb13d7dd91ab0e6431ebc2ccfc9c" ComfyUI-LLaVA-Captioner "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git "34bf73213d444f8270b6547eeb0d5982ce7ae50e" ComfyUI-Inspire-Pack "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/laksjdjf/attention-couple-ComfyUI.git "818b4a2af177d9578e86a5dbe5cab433be6f961c" attention-couple-ComfyUI "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/ExponentialML/ComfyUI_ELLA.git "303fe87f74c6237d05df8c17a06f2812e6028754" ComfyUI_ELLA "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/kijai/ComfyUI-KJNodes.git "0298e965d267977993715daa226fd2591d824ed8" ComfyUI-KJNodes "${TARGET_DIR}/custom_nodes"
checkout_gitrepo https://github.com/theUpsider/ComfyUI-Logic.git "ba753246ba7f11ca935a68b16c5ad83317edc089" ComfyUI-Logic "${TARGET_DIR}/custom_nodes"


echo "Downloading input files..."
# download_file 'https://photoshot-us.s3.amazonaws.com/comfy/logo-quick-qr-art_transparent_white_small.png' "${TARGET_DIR}/input/logo-quickqr.art-transparent.png"
# download_file 'https://purrfect-ai-use.s3.amazonaws.com/logo/text_art_logo.png' "${TARGET_DIR}/input/text_art_logo.png"
# download_file 'https://purrfect-ai-use.s3.amazonaws.com/logo/fusion_art_logo.png' "${TARGET_DIR}/input/fusion_art_logo.png" 
download_file 'https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors' "${TARGET_DIR}/custom_nodes/ComfyUI_IPAdapter_plus/models/ip-adapter-plus-face_sd15.safetensors"
download_file 'https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin' "${TARGET_DIR}/custom_nodes/ComfyUI_IPAdapter_plus/models/ip-adapter-faceid-plusv2_sd15.bin"
download_file 'https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors' "${TARGET_DIR}/custom_nodes/ComfyUI_IPAdapter_plus/models/ip-adapter-plus_sd15.safetensors"


if [ -z "${S3_PATH}" ]; then
    exit 0
fi
echo "Creating ${TARGET_FILE}..."
# tar gzip the folder and upload to S3
if [ -n "$(which pigz)" ]; then
    # use pigz to speed up compression on multiple cores
    tar -cv -C "${TARGET_DIR}" . | pigz -1 > "${TARGET_FILE}"
else
    # tar is slower
    tar -czvf ${TARGET_FILE} -C ${TARGET_DIR} .
fi
echo "Uploading ${S3_PATH}..."
aws s3 cp "${TARGET_FILE}" "${S3_PATH}"
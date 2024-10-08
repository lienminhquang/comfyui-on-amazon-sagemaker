echo "########################################"
echo "Init ComfyUI-Impack-Pack..."
cd /home/runner/ComfyUI/custom_nodes/ComfyUI-Impact-Pack
python3 -m install 

echo "########################################"
echo "Init ComfyUI-Allor..."
# copy config
cp /home/runner/custom_nodes/ComfyUI-Allor/config.json /home/runner/ComfyUI/custom_nodes/ComfyUI-Allor/config.json
echo "########################################"
docker run --rm -it --privileged --gpus=all -p 8501:8501 -p 5900:5900 -p 52000:22 -v C:\Development\Playground\dit\data:/data -v C:\Development\Playground\stable-diffusion:/sd -v C:\Users\Chen\.cache\huggingface\hub:/hf_hub --name docker_container_sd sd_docker
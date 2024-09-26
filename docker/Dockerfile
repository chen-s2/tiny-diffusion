FROM huggingface/transformers-pytorch-gpu

RUN apt-get update
RUN apt-get install -y openssh-server

RUN pip3 install accelerate
RUN pip3 install -i https://pypi.org/simple/ bitsandbytes
RUN pip3 install peft trl datasets
RUN pip3 install deepspeed
RUN pip3 install ninja
RUN pip3 install flash_attn
RUN pip3 install pynvml torchsummary tiktoken debugpy codecarbon streamlit

RUN apt-get install nano
RUN apt-get install net-tools

ENV HF_HOME='/hf_hub'
RUN echo "set tabsize 4" >> /etc/nanorc

WORKDIR /sd

RUN mkdir /var/run/sshd
RUN ssh-keygen -A

RUN echo 'root:admin' | chpasswd

RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Ensure SSH directory permissions are correct
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

EXPOSE 22

# Add a script to start the SSH service
RUN echo '#!/bin/bash\n\
service ssh start\n\
/usr/sbin/sshd\n\
/bin/bash' > /start.sh

# Make the start script executable
RUN chmod +x /start.sh

# Use the script as the container's command
CMD ["/start.sh"]
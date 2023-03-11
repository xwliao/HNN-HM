FROM tensorflow/tensorflow:2.3.0-gpu

RUN sed -i.bak \
        -e 's|http://.*ubuntu\.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|' \
        /etc/apt/sources.list \
    && apt-get update && apt-get install -y libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && mv /etc/apt/sources.list.bak /etc/apt/sources.list

RUN pip install -q \
        -i https://mirrors.aliyun.com/pypi/simple \
        --no-cache-dir \
        'tensorboard==2.3' \
        'tensorboard_plugin_profile==2.3' \
        'matplotlib==3.3.2' \
        'scikit-learn==0.23.2' \
        'scipy==1.4.1' \
        'pybind11==2.4.3' \
        'pytest==6.1.1' \
        'opencv-contrib-python==4.4.0.46' \
        'gdown==3.12.2' \
    && rm -rf $(pip cache dir)

#!/bin/sh

cd /opt/intel/openvino/deployment_tools/tools/model_downloader
sudo ./downloader.py --name {model-name} {--precisions {precision-levels}}
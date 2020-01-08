#!/bin/sh


cd SqueezeNet/SqueezeNet_v1.1/

C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat

python C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py 
    --input_model squeezenet_v1.1.caffemodel 
    --input_proto deploy.prototxt
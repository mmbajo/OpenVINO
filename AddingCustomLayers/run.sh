#!/bin/sh
export CLWS=/home/workspace/cl_tutorial
export CLT=$CLWS/OpenVINO-Custom-Layers

python /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample_async/classification_sample_async.py 
    -i $CLT/pics/dog.bmp -m $CLWS/cl_ext_cosh/model.ckpt.xml 
    -l $CLWS/cl_cosh/user_ie_extensions/cpu/build/libcosh_cpu_extension.so 
    -d CPU
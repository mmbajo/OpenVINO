#!/bin/sh

wget "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
tar -xvf "ssd_mobilenet_v2_coco_2018_03_29.tar.gz"

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py 
    --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb 
    --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config 
    --reverse_input_channels 
    --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
#!/bin/sh

python feed_network.py -m models/human-pose-estimation-0001.xml
python feed_network.py -m models/semantic-segmentation-adas-0001.xml
python feed_network.py -m models/text-detection-0004.xml
python feed_network.py -m models/vehicle-attributes-recognition-barrier-0039.xml
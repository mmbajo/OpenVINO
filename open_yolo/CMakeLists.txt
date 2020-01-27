# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME open_yolo
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS open_yolo.h
	      INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/opencv2" 
	      INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/opencv2/core"
	      OPENCV_DEPENDENCIES highgui
              DEPENDENCIES format_reader)

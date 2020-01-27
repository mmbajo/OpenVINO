// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample_async/main.cpp
* @example classification_sample_async/main.cpp
*/

#include <iostream>
#include <functional>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include <samples/ocv_common.hpp>
#include <opencv.hpp>
#include <core/mat.hpp>

#include <inference_engine.hpp>
#include <ie_core.hpp>

#include "open_yolo.h"
#include <ext_list.hpp>


using namespace InferenceEngine;

#define yolo_scale_13 13
#define yolo_scale_26 26
#define yolo_scale_52 52
#define DEVICE "CPU"
#define MODEL_FNAME "frozen_tiny_yolo_v3.xml"

// Define Helper Functions

void FrameToBlob(const cv::Mat& frame, InferRequest::Ptr& inferRequest, const std::string& inputName) {
	Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
	matU8ToBlob<uint8_t>(frame, frameBlob);
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) { // This function does ??????
	int n = location / (side * side);
	int loc = location % (side * side);
	return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

struct DetectionObject {
	int xmin, ymin, xmax, ymax, area, class_id;
	float confidence;

	DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
		this->xmin = static_cast<int>((x - w / 2) * w_scale);
		this->ymin = static_cast<int>((y - h / 2) * h_scale);
		this->xmax = static_cast<int>(this->xmin + w * w_scale);
		this->ymax = static_cast<int>(this->ymin + h * h_scale);
		this->area = static_cast<double>((this->ymax - this->ymin) * (this->xmax - this->xmin));
		this->class_id = class_id;
		this->confidence = confidence;
	}

	bool operator <(const DetectionObject& s2) const {
		return this->confidence < s2.confidence;
	}
	bool operator >(const DetectionObject& s2) const {
		return this->confidence > s2.confidence;
	}
};

double IntersectionOverUnion(const DetectionObject& box1, const DetectionObject& box2) {
	double widthOfOverlapArea = fmin(box1.xmax, box2.xmax) - fmax(box1.xmin, box2.xmin);
	double heightOfOverlapArea = fmin(box1.ymax, box2.ymax) - fmax(box1.ymin, box2.ymin);
	double areaOfOverlap;
	if (widthOfOverlapArea < 0 || heightOfOverlapArea < 0)
		areaOfOverlap = 0;
	else
		areaOfOverlap = widthOfOverlapArea * heightOfOverlapArea;
	double areaOfUnion = box1.area + box2.area - areaOfOverlap;
	return areaOfOverlap / areaOfUnion;
};

void ParseYOLOV3Output(const CNNLayerPtr& layer, const Blob::Ptr& blob, const unsigned long resized_im_h,
	const unsigned long resized_im_w, const unsigned long original_im_h,
	const unsigned long original_im_w,
	const double threshold, std::vector<DetectionObject>& objects) {
	// --------------------------- Validating output parameters -------------------------------------
	if (layer->type != "RegionYolo")
		throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo Expected");
	const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
	const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
	if (out_blob_h != out_blob_w)
		throw std::runtime_error("Invalid size of output " + layer->name +
			" It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
			", current W = " + std::to_string(out_blob_h));

	// --------------------------- Extracting layer parameters -------------------------------------
	auto num = layer->GetParamAsInt("num");
	try { num = layer->GetParamAsInts("mask").size(); }
	catch (...) {}
	auto coords = layer->GetParamAsInt("coords");
	auto classes = layer->GetParamAsInt("classes");
	std::vector<float> anchors = { 10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 }; // Research this shit
	try { anchors = layer->GetParamAsFloats("anchors"); }
	catch (...) {} // If there are user defined anchors get it
	auto side = out_blob_h;
	int anchor_offset = 0;

	if (anchors.size() == 18) {
		switch (side) {
		case yolo_scale_13:
			anchor_offset = 2 * 6;
			break;
		case yolo_scale_26:
			anchor_offset = 2 * 3;
			break;
		case yolo_scale_52:
			anchor_offset = 2 * 0;
			break;
		default:
			throw std::runtime_error("Invalid output size");
		}
	}
	else if (anchors.size() == 12) {
		switch (side) {
		case yolo_scale_13:
			anchor_offset = 2 * 3;
			break;
		case yolo_scale_26:
			anchor_offset = 2 * 0;
			break;
		default:
			throw std::runtime_error("Invalid output size");
		}
	}
	else {
		switch (side) {
		case yolo_scale_13:
			anchor_offset = 2 * 6;
			break;
		case yolo_scale_26:
			anchor_offset = 2 * 3;
			break;
		case yolo_scale_52:
			anchor_offset = 2 * 0;
			break;
		default:
			throw std::runtime_error("Invalid output size");
		}
	}
	auto side_square = side * side;
	const float* output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>(); // this is where you set the model precision -> for faster inference time set FP32 to INT8

	// --------------------------- Parsing YOLO Region output -------------------------------------
	for (int i = 0; i < side_square; ++i) {
		int row = i / side;
		int col = i % side;
		for (int n = 0; n < num; ++n) {
			int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
			int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
			float scale = output_blob[obj_index];
			if (scale < threshold)
				continue;
			double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
			double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
			double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1];
			double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n];
			for (int j = 0; j < classes; ++j) {
				int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
				float prob = scale * output_blob[class_index];
				if (prob < threshold)
					continue;
				DetectionObject obj(x, y, height, width, j, prob,
					static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
					static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
				objects.push_back(obj); // Collect obj and dumpt to vector objects
			}
		}
	}
}


void detect(cv::Mat frame, double obj_conf_threshold, double iou_threshold) {
	// Extract Constants
	const int width = frame.cols;
	const int height = frame.rows;

	// --------------------------- 1. Load inference engine -------------------------------------
	std::cout << "Load inference engine!" << std::endl;
	Core ie;
	std::cout << "Loaded inference engine!" << std::endl;
	//std::cout << ie.GetVersions("CPU");


	/** Loading exttensions to the plugin **/

	/** Loading default extensions **/
	// Should i?


	// --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
	std::cout << "Load Model!" << std::endl;
	CNNNetReader netReader;
	/** Read network model **/
	netReader.ReadNetwork(MODEL_FNAME);
	/** Set batch size to 1 **/
	netReader.getNetwork().setBatchSize(1);
	/** Extracting the model name and loading its weights **/
	std::string binFileName = fileNameNoExt(MODEL_FNAME) + ".bin";
	netReader.ReadWeights(binFileName);
	/** Reading labels **/
	std::string labelFileName = fileNameNoExt(MODEL_FNAME) + ".labels";
	std::vector<std::string> labels;
	std::ifstream inputFile(labelFileName);
	std::copy(std::istream_iterator<std::string>(inputFile),
		std::istream_iterator < std::string>(),
		std::back_inserter(labels));
	std::cout << "Loaded Model!" << std::endl;
	// -----------------------------------------------------------------------------------------------------

	/** YOLOV3-based network should have one input and three output **/
	// --------------------------- 3. Configuring input and output -----------------------------------------
	// --------------------------------- Preparing input blobs ---------------------------------------------
	std::cout << "Prepare Input Blob!" << std::endl;
	InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
	if (inputInfo.size() != 1) {
		throw std::logic_error("Wrong input size! Must be value of 1.");
	}
	InputInfo::Ptr& input = inputInfo.begin()->second;
	auto inputName = inputInfo.begin()->first;
	input->setPrecision(Precision::U8); // Should i change this??
	input->getInputData()->setLayout(Layout::NCHW);
	std::cout << "Input Blob Preparation Success!" << std::endl;
	// --------------------------------- Preparing output blobs -------------------------------------------
	std::cout << "Prepare Output Blobs!" << std::endl;
	OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
	for (auto& output : outputInfo) {
		output.second->setPrecision(Precision::FP16);
		output.second->setLayout(Layout::NCHW);
	}
	std::cout << "Output Blobs preparation success!" << std::endl;
	// -----------------------------------------------------------------------------------------------------

	// --------------------------- 4. Loading model to the device ------------------------------------------
	std::cout << "Loading Model to the Device!" << std::endl;
	ExecutableNetwork network = ie.LoadNetwork(netReader.getNetwork(), DEVICE);
	std::cout << "Loading Successful!" << std::endl;

	// -----------------------------------------------------------------------------------------------------

	// --------------------------- 5. Creating infer request -----------------------------------------------
	InferRequest::Ptr infer_request = network.CreateInferRequestPtr();
	// -----------------------------------------------------------------------------------------------------

	// --------------------------- 6. Doing inference ------------------------------------------------------
	FrameToBlob(frame, infer_request, inputName);

	// ---------------------------Processing output blobs--------------------------------------------------
	const TensorDesc& inputDesc = inputInfo.begin()->second.get()->getTensorDesc();
	unsigned long resized_im_h = getTensorHeight(inputDesc);
	unsigned long resized_im_w = getTensorWidth(inputDesc);
	std::vector<DetectionObject> objects;

	// Parse Outputs
	for (auto& output : outputInfo) {
		auto output_name = output.first;
		CNNLayerPtr layer = netReader.getNetwork().getLayerByName(output_name.c_str());
		Blob::Ptr blob = infer_request->GetBlob(output_name);
		ParseYOLOV3Output(layer, blob, resized_im_h, resized_im_w, height, width, obj_conf_threshold, objects);
	}
	// Filter overlapping boxes
	std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
	for (size_t i = 0; i < objects.size(); ++i) {
		if (objects[i].confidence == 0)
			continue;
		for (size_t j = i + 1; j < objects.size(); ++j)
			if (IntersectionOverUnion(objects[i], objects[j]) >= iou_threshold)
				objects[j].confidence = 0;
	}
	// Draw Boxes
	std::cout << "Drawing Boxes";
	for (auto& object : objects) {
		if (object.confidence < obj_conf_threshold)
			continue;
		auto label = object.class_id;
		float confidence = object.confidence;

		std::ostringstream conf;
		conf << ":" << std::fixed << std::setprecision(3) << confidence;
		cv::putText(frame,
			(label < static_cast<int>(labels.size()) ?
				labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
			cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
			cv::Scalar(0, 0, 255));
		cv::rectangle(frame, cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin)),
			cv::Point2f(static_cast<float>(object.xmax), static_cast<float>(object.ymax)), cv::Scalar(0, 0, 255));
	}
	// Show pic
	cv::imshow("Detection results", frame);
	cv::imwrite("kuting_out.jpg", frame);
}

int main() {
	std::cout << "Hello World!" << std::endl;
	cv::Mat frame = cv::imread("kuting.jpg");
	detect(frame, 0.5, 0.3);
	return 0;
}

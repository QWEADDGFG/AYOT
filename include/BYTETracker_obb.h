#pragma once

#include "STrack_obb.h"
#include <opencv2/opencv.hpp>

struct Object_OBB
{
	cv::RotatedRect rect;
    int label;
    float prob;
};

class BYTETracker_obb
{
public:
	BYTETracker_obb(int frame_rate = 30, int track_buffer = 20);
	~BYTETracker_obb();

	std::vector<STrack_obb> update(const std::vector<Object_OBB>& objects);
	cv::Scalar get_color(int idx);

private:
	std::vector<STrack_obb*> joint_stracks(std::vector<STrack_obb*> &tlista, std::vector<STrack_obb> &tlistb);
	std::vector<STrack_obb> joint_stracks(std::vector<STrack_obb> &tlista, std::vector<STrack_obb> &tlistb);

	std::vector<STrack_obb> sub_stracks(std::vector<STrack_obb> &tlista, std::vector<STrack_obb> &tlistb);
	void remove_duplicate_stracks(std::vector<STrack_obb> &resa, std::vector<STrack_obb> &resb, std::vector<STrack_obb> &stracksa, std::vector<STrack_obb> &stracksb);

	void linear_assignment(std::vector<std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		std::vector<std::vector<int> > &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b);
	std::vector<std::vector<float> > iou_distance(std::vector<STrack_obb*> &atracks, std::vector<STrack_obb> &btracks, int &dist_size, int &dist_size_size);
	std::vector<std::vector<float> > iou_distance(std::vector<STrack_obb> &atracks, std::vector<STrack_obb> &btracks);
	std::vector<std::vector<float> > ious(std::vector<std::vector<float> > &atlbrs, std::vector<std::vector<float> > &btlbrs);

	double lapjv(const std::vector<std::vector<float> > &cost, std::vector<int> &rowsol, std::vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh;
	float high_thresh;
	float match_thresh;
	int frame_id;
	int max_time_lost;

	std::vector<STrack_obb> tracked_stracks;
	std::vector<STrack_obb> lost_stracks;
	std::vector<STrack_obb> removed_stracks;
	byte_kalman::KalmanFilter_obb kalman_filter;
};
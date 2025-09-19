#pragma once

#include "kalmanFilter_obb.h"

//using namespace cv;
//using namespace std;

enum TrackState_OBB { New = 0, Tracked, Lost, Removed };

class STrack_obb
{
public:
	STrack_obb(std::vector<float> tlwh_, float score, int cls);
	~STrack_obb();

	//std::vector<float> static tlbr_to_tlwh(std::vector<float> &tlbr);
	void static multi_predict(std::vector<STrack_obb*> &stracks, byte_kalman::KalmanFilter_obb &kalman_filter);
	void static_tlwh();
	//void static_tlbr();
	std::vector<float> tlwh_to_xyah(std::vector<float> tlwh_tmp);
	std::vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();
	
	void activate(byte_kalman::KalmanFilter_obb &kalman_filter, int frame_id);
	void re_activate(STrack_obb &new_track, int frame_id, bool new_id = false);
	void update(STrack_obb &new_track, int frame_id);

public:
	bool is_activated;
	int track_id;
	int state;

	std::vector<float> _tlwh;
	std::vector<float> tlwh; //(cx, cy, w, h, angle)
	//std::vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;

	KAL_MEAN_OBB mean;
	KAL_COVA_OBB covariance;
	float score;
	int cls;

private:
	byte_kalman::KalmanFilter_obb kalman_filter;
};
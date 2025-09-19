#include "STrack_obb.h"

STrack_obb::STrack_obb(std::vector<float> tlwh_, float score, int cls)
{
	_tlwh.resize(5);
	_tlwh.assign(tlwh_.begin(), tlwh_.end());

	is_activated = false;
	track_id = 0;
	state = TrackState_OBB::New;
	
	tlwh.resize(5);
	//tlbr.resize(5);

	static_tlwh();
	//static_tlbr();
	frame_id = 0;
	tracklet_len = 0;
	this->score = score;
	this->cls = cls;
	start_frame = 0;
}

STrack_obb::~STrack_obb()
{
}

void STrack_obb::activate(byte_kalman::KalmanFilter_obb &kalman_filter, int frame_id)
{
	this->kalman_filter = kalman_filter;
	this->track_id = this->next_id();

	std::vector<float> _tlwh_tmp(5);
	_tlwh_tmp[0] = this->_tlwh[0];
	_tlwh_tmp[1] = this->_tlwh[1];
	_tlwh_tmp[2] = this->_tlwh[2];
	_tlwh_tmp[3] = this->_tlwh[3];
	_tlwh_tmp[4] = this->_tlwh[4];
	std::vector<float> xyah = tlwh_to_xyah(_tlwh_tmp);
	DETECTBOX_OBB xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	xyah_box[4] = xyah[4];
	auto mc = this->kalman_filter.initiate(xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	//static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState_OBB::Tracked;
	if (frame_id == 1)
	{
		this->is_activated = true;
	}
	//this->is_activated = true;
	this->frame_id = frame_id;
	this->start_frame = frame_id;
}

void STrack_obb::re_activate(STrack_obb &new_track, int frame_id, bool new_id)
{
	std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX_OBB xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	xyah_box[4] = xyah[4];
	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	//static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState_OBB::Tracked;
	this->is_activated = true;
	this->frame_id = frame_id;
	this->score = new_track.score;
	this->cls = new_track.cls;
	if (new_id)
		this->track_id = next_id();
}

void STrack_obb::update(STrack_obb &new_track, int frame_id)
{
	this->frame_id = frame_id;
	this->tracklet_len++;

	std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX_OBB xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	xyah_box[4] = xyah[4];

	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	//static_tlbr();

	this->state = TrackState_OBB::Tracked;
	this->is_activated = true;

	this->score = new_track.score;
}

void STrack_obb::static_tlwh()
{
	if (this->state == TrackState_OBB::New)
	{
		tlwh[0] = _tlwh[0];
		tlwh[1] = _tlwh[1];
		tlwh[2] = _tlwh[2];
		tlwh[3] = _tlwh[3];
		tlwh[4] = _tlwh[4];
		return;
	}

	tlwh[0] = mean[0];
	tlwh[1] = mean[1];
	tlwh[2] = mean[2];
	tlwh[3] = mean[3];
	tlwh[4] = mean[4];

	tlwh[2] *= tlwh[3];
	//tlwh[0] -= tlwh[2] / 2;
	//tlwh[1] -= tlwh[3] / 2;
}

//void STrack_obb::static_tlbr()
//{
//	tlbr.clear();
//	tlbr.assign(tlwh.begin(), tlwh.end());
//	tlbr[2] += tlbr[0];
//	tlbr[3] += tlbr[1];
//}

std::vector<float> STrack_obb::tlwh_to_xyah(std::vector<float> tlwh_tmp)
{
	std::vector<float> tlwh_output = tlwh_tmp;
	//tlwh_output[0] += tlwh_output[2] / 2;
	//tlwh_output[1] += tlwh_output[3] / 2;
	tlwh_output[2] /= tlwh_output[3];
	return tlwh_output;
}

std::vector<float> STrack_obb::to_xyah()
{
	return tlwh_to_xyah(tlwh);
}

//std::vector<float> STrack_obb::tlbr_to_tlwh(std::vector<float> &tlbr)
//{
//	tlbr[2] -= tlbr[0];
//	tlbr[3] -= tlbr[1];
//	return tlbr;
//}

void STrack_obb::mark_lost()
{
	state = TrackState_OBB::Lost;
}

void STrack_obb::mark_removed()
{
	state = TrackState_OBB::Removed;
}

int STrack_obb::next_id()
{
	static int _count = 0;
	_count++;
	return _count;
}

int STrack_obb::end_frame()
{
	return this->frame_id;
}

void STrack_obb::multi_predict(std::vector<STrack_obb*> &stracks, byte_kalman::KalmanFilter_obb &kalman_filter)
{
	for (int i = 0; i < stracks.size(); i++)
	{
		if (stracks[i]->state != TrackState_OBB::Tracked)
		{
			stracks[i]->mean[8] = 0; //7
		}
		kalman_filter.predict(stracks[i]->mean, stracks[i]->covariance);
		stracks[i]->static_tlwh();
		//stracks[i]->static_tlbr();
	}
}
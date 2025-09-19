#include "BYTETracker_obb.h"
#include <fstream>

BYTETracker_obb::BYTETracker_obb(int frame_rate, int track_buffer)
{
	track_thresh = 0.15f; //用于判断检测结果的低分还是高分
	high_thresh = 0.25f;  //用于跟踪起始的检测阈值
	match_thresh = 0.9f;

	frame_id = 0;
	max_time_lost = int(frame_rate / 30.0 * track_buffer);
	std::cout << "Init ByteTrack!" << std::endl;
}

BYTETracker_obb::~BYTETracker_obb()
{
}

std::vector<STrack_obb> BYTETracker_obb::update(const std::vector<Object_OBB>& objects)
{

	////////////////// Step 1: Get detections //////////////////
	this->frame_id++;
	std::vector<STrack_obb> activated_stracks;
	std::vector<STrack_obb> refind_stracks;   //lost_track 重新跟踪
	std::vector<STrack_obb> removed_stracks;  //移除跟踪结果
	std::vector<STrack_obb> lost_stracks;     //丢失跟踪结果
	std::vector<STrack_obb> detections;       //高分检测结果
	std::vector<STrack_obb> detections_low;   //低分检测结果

	std::vector<STrack_obb> detections_cp;    //未匹配高分检测结果
	std::vector<STrack_obb> tracked_stracks_swap;
	std::vector<STrack_obb> resa, resb;
	std::vector<STrack_obb> output_stracks;

	std::vector<STrack_obb*> unconfirmed;     //未确认跟踪结果
	std::vector<STrack_obb*> tracked_stracks; //已跟踪上结果
	std::vector<STrack_obb*> strack_pool;     //临时变量
	std::vector<STrack_obb*> r_tracked_stracks; //未匹配上track与低分检测匹配的临时变量

	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			std::vector<float> tlbr_;
			tlbr_.resize(5);
			tlbr_[0] = objects[i].rect.center.x;
			tlbr_[1] = objects[i].rect.center.y;
			tlbr_[2] = objects[i].rect.size.width;
			tlbr_[3] = objects[i].rect.size.height;
			tlbr_[4] = objects[i].rect.angle;

			float score = objects[i].prob;
			int cls = objects[i].label;

			STrack_obb strack(tlbr_, score, cls);
			if (score >= track_thresh)
			{
				detections.push_back(strack);  //高分
			}
			else
			{
				detections_low.push_back(strack); //低分
			}
		}
	}

	// Add newly detected tracklets to tracked_stracks
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (!this->tracked_stracks[i].is_activated)
			unconfirmed.push_back(&this->tracked_stracks[i]);     //未确认
		else
			tracked_stracks.push_back(&this->tracked_stracks[i]); //已跟踪结果
	}

	////////////////// Step 2: First association, with IoU //////////////////
	strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
	STrack_obb::multi_predict(strack_pool, this->kalman_filter);

	std::vector<std::vector<float> > dists;
	int dist_size = 0, dist_size_size = 0;
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

	std::vector<std::vector<int> > matches;
	std::vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

	//匹配上进行跟踪，或者重新激活
	for (int i = 0; i < matches.size(); i++)
	{
		STrack_obb *track = strack_pool[matches[i][0]];
		STrack_obb *det = &detections[matches[i][1]];
		if (track->state == TrackState_OBB::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else  //lost
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	////////////////// Step 3: Second association, using low score dets //////////////////
	for (int i = 0; i < u_detection.size(); i++)
	{
		detections_cp.push_back(detections[u_detection[i]]);
	}
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());
	
	for (int i = 0; i < u_track.size(); i++)
	{
		if (strack_pool[u_track[i]]->state == TrackState_OBB::Tracked)
		{
			r_tracked_stracks.push_back(strack_pool[u_track[i]]);
		}
	}

	dists.clear();
	//未匹配跟踪结果与低分检测结果进行二次匹配
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

	matches.clear();
	u_track.clear();
	u_detection.clear();
	//阈值低
	linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		STrack_obb *track = r_tracked_stracks[matches[i][0]];
		STrack_obb *det = &detections[matches[i][1]];
		if (track->state == TrackState_OBB::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	//仍未匹配的跟踪结果转为lost状态
	for (int i = 0; i < u_track.size(); i++)
	{
		STrack_obb *track = r_tracked_stracks[u_track[i]];
		if (track->state != TrackState_OBB::Lost)
		{
			track->mark_lost();
			lost_stracks.push_back(*track);
		}
	}

	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	//与未确认的跟踪进行匹配
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	std::vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}

	//删除
	for (int i = 0; i < u_unconfirmed.size(); i++)
	{
		STrack_obb *track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}

	////////////////// Step 4: Init new stracks //////////////////
	//最终与activated， lost, unconfirmed均为匹配的高分检测进行初始化
	for (int i = 0; i < u_detection.size(); i++)
	{
		STrack_obb *track = &detections[u_detection[i]];
		if (track->score < this->high_thresh)
			continue;
		track->activate(this->kalman_filter, this->frame_id);
		activated_stracks.push_back(*track);
	}

	////////////////// Step 5: Update state //////////////////
	for (int i = 0; i < this->lost_stracks.size(); i++)
	{
		if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost)
		{
			this->lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->lost_stracks[i]);
		}
	}
	
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].state == TrackState_OBB::Tracked)
		{
			tracked_stracks_swap.push_back(this->tracked_stracks[i]);
		}
	}
	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

	//std::cout << activated_stracks.size() << std::endl;

	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (int i = 0; i < lost_stracks.size(); i++)
	{
		this->lost_stracks.push_back(lost_stracks[i]);
	}

	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
	for (int i = 0; i < removed_stracks.size(); i++)
	{
		this->removed_stracks.push_back(removed_stracks[i]);
	}
	
	remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(resa.begin(), resa.end());
	this->lost_stracks.clear();
	this->lost_stracks.assign(resb.begin(), resb.end());
	
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].is_activated)
		{
			output_stracks.push_back(this->tracked_stracks[i]);
		}
	}
	return output_stracks;
}
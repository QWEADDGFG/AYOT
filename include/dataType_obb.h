#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
//x, y, aspectratio, h, angle
typedef Eigen::Matrix<float, 1, 5, Eigen::RowMajor> DETECTBOX_OBB;
typedef Eigen::Matrix<float, -1, 5, Eigen::RowMajor> DETECTBOXSS_OBB;
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE_OBB;
typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> FEATURESS_OBB;
//typedef std::vector<FEATURE> FEATURESS;

//Kalmanfilter
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
//x, y, aspectratio, h, angle, vx, vy, va, vh, vangle
typedef Eigen::Matrix<float, 1, 10, Eigen::RowMajor> KAL_MEAN_OBB; //按行存储
typedef Eigen::Matrix<float, 10, 10, Eigen::RowMajor> KAL_COVA_OBB;
typedef Eigen::Matrix<float, 1, 5, Eigen::RowMajor> KAL_HMEAN_OBB;
typedef Eigen::Matrix<float, 5, 5, Eigen::RowMajor> KAL_HCOVA_OBB;
using KAL_DATA_OBB = std::pair<KAL_MEAN_OBB, KAL_COVA_OBB>;
using KAL_HDATA_OBB = std::pair<KAL_HMEAN_OBB, KAL_HCOVA_OBB>;

//main
using RESULT_DATA_OBB = std::pair<int, DETECTBOX_OBB>;

//tracker:
using TRACKER_DATA_OBB = std::pair<int, FEATURESS_OBB>;
using MATCH_DATA_OBB = std::pair<int, int>;
typedef struct t {
	std::vector<MATCH_DATA_OBB> matches;
	std::vector<int> unmatched_tracks;
	std::vector<int> unmatched_detections;
}TRACHER_MATCHD_OBB;

//linear_assignment:
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM_OBB;

// #define M_PI 3.1415926
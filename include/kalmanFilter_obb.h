#pragma once

#include "dataType_obb.h"

namespace byte_kalman
{
	class KalmanFilter_obb
	{
	public:
		static const double chi2inv95[10];
		KalmanFilter_obb();
		KAL_DATA_OBB initiate(const DETECTBOX_OBB& measurement);
		void predict(KAL_MEAN_OBB& mean, KAL_COVA_OBB& covariance);
		KAL_HDATA_OBB project(const KAL_MEAN_OBB& mean, const KAL_COVA_OBB& covariance);
		KAL_DATA_OBB update(const KAL_MEAN_OBB& mean,
			const KAL_COVA_OBB& covariance,
			const DETECTBOX_OBB& measurement);

		Eigen::Matrix<float, 1, -1> gating_distance(
			const KAL_MEAN_OBB& mean,
			const KAL_COVA_OBB& covariance,
			const std::vector<DETECTBOX_OBB>& measurements,
			bool only_position = false);

	private:
		Eigen::Matrix<float, 10, 10, Eigen::RowMajor> _motion_mat;
		Eigen::Matrix<float, 5, 10, Eigen::RowMajor> _update_mat;
		float _std_weight_position;
		float _std_weight_velocity;
		float _std_weight_angle;
		//cx, cy, w, h, angle
	};
}
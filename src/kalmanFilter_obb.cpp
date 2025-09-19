#include "kalmanFilter_obb.h"
#include <Eigen/Cholesky>

namespace byte_kalman
{
	const double KalmanFilter_obb::chi2inv95[10] = {
	0,
	3.8415,
	5.9915,
	7.8147,
	9.4877,
	11.070,
	12.592,
	14.067,
	15.507,
	16.919
	};
	KalmanFilter_obb::KalmanFilter_obb()
	{
		int ndim = 5;
		double dt = 1.;

		_motion_mat = Eigen::MatrixXf::Identity(10, 10);
		for (int i = 0; i < ndim; i++) {
			_motion_mat(i, ndim + i) = dt;
		}
		_update_mat = Eigen::MatrixXf::Identity(5, 10);

		this->_std_weight_position = 1. / 20;
		this->_std_weight_velocity = 1. / 160;
		this->_std_weight_angle = 1. / 160; //如何设置，角度有周期性
	}

	KAL_DATA_OBB KalmanFilter_obb::initiate(const DETECTBOX_OBB &measurement)
	{
		//x, y, a, h, theta
		DETECTBOX_OBB mean_pos = measurement;
		DETECTBOX_OBB mean_vel;
		for (int i = 0; i < 5; i++) mean_vel(i) = 0;

		KAL_MEAN_OBB mean;
		for (int i = 0; i < 10; i++) {
			if (i < 5) mean(i) = mean_pos(i);                //位置为测量值
			else mean(i) = mean_vel(i - 5);                  //速度为0
		}

		KAL_MEAN_OBB std;
		std(0) = 2 * _std_weight_position * measurement[3];  //x
		std(1) = 2 * _std_weight_position * measurement[3];  //y
		std(2) = 1e-2;                                       //aspectratio
		std(3) = 2 * _std_weight_position * measurement[3];  //h
		std(4) = 1 * _std_weight_angle;     //angle
		std(5) = 10 * _std_weight_velocity * measurement[3]; //vx
		std(6) = 10 * _std_weight_velocity * measurement[3]; //vy
		std(7) = 1e-5;                                       //va
		std(8) = 10 * _std_weight_velocity * measurement[3]; //vh
		std(9) = 5 * _std_weight_angle;     //vangle 

		KAL_MEAN_OBB tmp = std.array().square();
		KAL_COVA_OBB var = tmp.asDiagonal();
		return std::make_pair(mean, var);
	}
	//https://blog.csdn.net/weixin_45539933/article/details/150396903
	void KalmanFilter_obb::predict(KAL_MEAN_OBB &mean, KAL_COVA_OBB &covariance)
	{
		//revise the data;
		DETECTBOX_OBB std_pos;
		//过程噪声协方差矩阵，描述系统模型的不确定性（如路面颠簸导致的运动误差）。 Q
		std_pos << _std_weight_position * mean(3),
			_std_weight_position * mean(3),
			1e-2,
			_std_weight_position * mean(3),
			_std_weight_angle; //?
		DETECTBOX_OBB std_vel;
		std_vel << _std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			1e-5,
			_std_weight_velocity * mean(3),
			_std_weight_angle; //?
		KAL_MEAN_OBB tmp;
		//P.block<rows, cols>(i, j)          // P(i+1 : i+rows, j+1 : j+cols) i,j开始，rows行cols列
		tmp.block<1, 5>(0, 0) = std_pos;
		tmp.block<1, 5>(0, 5) = std_vel;
		tmp = tmp.array().square();
		KAL_COVA_OBB motion_cov = tmp.asDiagonal();
		//一步预测
		KAL_MEAN_OBB mean1 = this->_motion_mat * mean.transpose(); //？没加噪声项x(k)=A*x(k-1)+u(k)
		//预测误差协方差方程（不确定性预测）P(k) = A*P(k-1)*A^t+Q
		KAL_COVA_OBB covariance1 = this->_motion_mat * covariance *(_motion_mat.transpose());
		covariance1 += motion_cov;

		mean = mean1;
		covariance = covariance1;
	}

	KAL_HDATA_OBB KalmanFilter_obb::project(const KAL_MEAN_OBB &mean, const KAL_COVA_OBB &covariance)
	{
		DETECTBOX_OBB std;
		//观测协方差矩阵，R
		std << _std_weight_position * mean(3), 
			_std_weight_position * mean(3),
			1e-1, 
			_std_weight_position * mean(3), 
			_std_weight_angle * mean(3); //?
		//观测预测
		KAL_HMEAN_OBB mean1 = _update_mat * mean.transpose();
		//H(k)*P(k)*H(k)^t+R
		KAL_HCOVA_OBB covariance1 = _update_mat * covariance * (_update_mat.transpose());
		Eigen::Matrix<float, 5, 5> diag = std.asDiagonal();
		diag = diag.array().square().matrix();
		covariance1 += diag;
		//    covariance1.diagonal() << diag;
		return std::make_pair(mean1, covariance1);
	}

	KAL_DATA_OBB
		KalmanFilter_obb::update(
			const KAL_MEAN_OBB &mean,
			const KAL_COVA_OBB &covariance,
			const DETECTBOX_OBB &measurement)
	{
		KAL_HDATA_OBB pa = project(mean, covariance);
		KAL_HMEAN_OBB projected_mean = pa.first;
		KAL_HCOVA_OBB projected_cov = pa.second;

		//chol_factor, lower =
		//scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
		//kalmain_gain =
		//scipy.linalg.cho_solve((cho_factor, lower),
		//np.dot(covariance, self._upadte_mat.T).T,
		//check_finite=False).T
		Eigen::Matrix<float, 5, 10> B = (covariance * (_update_mat.transpose())).transpose();
		//卡尔曼增益K=P(k)*H^t*projected_cov^-1=B^-1*projected_cov
		Eigen::Matrix<float, 10, 5> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.10x5
		//新息
		Eigen::Matrix<float, 1, 5> innovation = measurement - projected_mean; //eg.1x5
		auto tmp = innovation * (kalman_gain.transpose());
		//预测校正， 1*10
		KAL_MEAN_OBB new_mean = (mean.array() + tmp.array()).matrix();
		//需要处理，角度周期性问题，在角度边界处的突变如何处理，以及如何进行限位, [-PI/2, PI/2]
		if (new_mean(0, 4) * measurement(0, 4) < 0)
		{
			new_mean(0, 4) = measurement(0, 4);
			new_mean(0, 9) = 1e-3f;
		}
		
		//过程噪声协方差矩阵更新P(k)=P(k-1)-K*H*P(k-1)
		KAL_COVA_OBB new_covariance = covariance - kalman_gain * projected_cov*(kalman_gain.transpose());
		return std::make_pair(new_mean, new_covariance);
	}

	Eigen::Matrix<float, 1, -1>
		KalmanFilter_obb::gating_distance(
			const KAL_MEAN_OBB &mean,
			const KAL_COVA_OBB &covariance,
			const std::vector<DETECTBOX_OBB> &measurements,
			bool only_position)
	{
		KAL_HDATA_OBB pa = this->project(mean, covariance);
		if (only_position) {
			printf("not implement!");
			exit(0);
		}
		KAL_HMEAN_OBB mean1 = pa.first;
		KAL_HCOVA_OBB covariance1 = pa.second;

		//    Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
		DETECTBOXSS_OBB d(measurements.size(), 5);
		int pos = 0;
		for (DETECTBOX_OBB box : measurements) {
			d.row(pos++) = box - mean1;
		}
		Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
		Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
		auto zz = ((z.array())*(z.array())).matrix();
		auto square_maha = zz.colwise().sum();
		return square_maha;
	}
}
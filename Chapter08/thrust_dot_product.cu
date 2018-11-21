#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

using namespace std;

struct multiply_functor {

	float w;

	multiply_functor(float _w = 1) : w(_w) {}

	__device__ float operator() (const float & x, const float & y)  { 
		return  w * x * y;
	}
};

float dot_product(thrust::device_vector<float> &v, thrust::device_vector<float> &w ) //, thrust::device_vector<float> &z)
{
	thrust::device_vector<float> z(v.size());

	thrust::transform(v.begin(), v.end(), w.begin(), z.begin(), multiply_functor());
	
	return thrust::reduce(z.begin(), z.end());
}

int main(void)
{

	thrust::device_vector<float> v;

	v.push_back(1.0f);
	v.push_back(2.0f);
	v.push_back(3.0f);

	thrust::device_vector<float> w(3);

	thrust::fill(w.begin(), w.end(), 1.0f);

	for (int i = 0; i < v.size(); i++)
		cout << "v[" << i << "] == " << v[i] << endl;

	for (int i = 0; i < w.size(); i++)
		cout << "w[" << i << "] == " << w[i] << endl;
	
	cout << "dot_product(v , w) == " << dot_product(v,w) << endl;

	return 0;
}

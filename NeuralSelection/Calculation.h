#pragma once

#include <math.h>

float sigmoid(float x)
{
	return 1.0f / (1.0f + powf(2.71828f, -x));
}

void softmax(float *results, float *values, const size_t count, const size_t setCount)
{
	float *temp = new float[count];
	float *index = values;
	float sum;

	for (size_t i = 0; i < setCount; i++, index += 3)
	{
		sum = 0.0f;

		for (size_t n = 0; n < count; n++)
		{
			temp[n] = expf(index[n]);
			sum += temp[n];
		}

		for (size_t n = 0; n < count; n++)
		{
			results[i * count + n] = (temp[n] / sum);
		}
	}
}

float* softmax(float *values, const size_t count, const size_t setCount)
{
	float *r = new float[setCount * count];
	
	softmax(r, values, count, setCount);

	return r;
}

float sum(const float *values, const size_t count)
{
	float sum = 0.0f;

	for (size_t i = 0; i < count; i++)
	{
		sum += values[i];
	}

	return sum;
}
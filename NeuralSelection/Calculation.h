#pragma once

#include <math.h>

float sigmoid(float x)
{
	return 1.0f / (1.0f + powf(2.71828f, -x));
}
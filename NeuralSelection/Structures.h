#pragma once

#include <string>
#include <stdint.h>
#include <vector>
#include "Calculation.h"

using namespace std;

union FastFloat
{
	uint64_t i;
	float x;
};

struct Xor1024
{
	uint64_t s1024[16];
	int p;
};

struct StockData
{
	uint32_t Date, Volume;
	float Open, High, Low, Close, AveragePrice;

	StockData() {}

	StockData(uint32_t date, float open, float high, float low, float close, uint32_t volume)
	{
		this->Date = date;
		this->Open = open;
		this->High = high;
		this->Low = low;
		this->Close = close;
		this->Volume = volume;
		this->AveragePrice = (this->High + this->Low) / 2.0f;
	}
};

struct StockDataVector
{
	string Description;
	vector<StockData> *Data;

	StockDataVector()
	{
		this->Data = new vector<StockData>();
	}

	StockDataVector(vector<StockData> *v)
	{
		this->Data = v;
	
	}

	~StockDataVector()
	{		
		delete this->Data;

		this->Data = NULL;
	}
};

struct SimpleNeuralNetwork
{
	size_t Predictors, HiddenUnits, OutputUnits;
	float *InputToHiddenWeights;
	float *HiddenToOutputWeights;
	float *HiddenBiasWeights;
	float *OutputBiasWeights;

	SimpleNeuralNetwork(size_t predictors, size_t hiddenUnits, size_t outputUnits)
	{
		this->Predictors = predictors;
		this->HiddenUnits = hiddenUnits;
		this->OutputUnits = outputUnits;
		this->InputToHiddenWeights = new float[predictors * hiddenUnits];
		this->HiddenToOutputWeights = new float[hiddenUnits * outputUnits];
		this->HiddenBiasWeights = new float[hiddenUnits];
		this->OutputBiasWeights = new float[outputUnits];
	}

	float* CreateHiddenResultSet() {
		return new float[this->HiddenUnits];
	}

	float* CreateOutputResultSet() {
		return new float[this->OutputUnits];
	}

	void CalculateRaw(float *inputs, float *hiddenResults, float *outputResults)
	{
		float result;
		size_t startIndex;

		for (size_t hiddenNeuron = 0; hiddenNeuron < this->HiddenUnits; hiddenNeuron++)
		{
			result = this->HiddenBiasWeights[hiddenNeuron];
			startIndex = hiddenNeuron * this->Predictors;

			for (size_t i = 0; i < this->Predictors; i++)
			{
				result += inputs[i] * this->InputToHiddenWeights[startIndex + i];
			}

			hiddenResults[hiddenNeuron] = result;
		}

		for (size_t outputNeuron = 0; outputNeuron < this->OutputUnits; outputNeuron++)
		{
			result = this->OutputBiasWeights[outputNeuron];
			startIndex = outputNeuron * this->HiddenUnits;

			for (size_t i = 0; i < this->HiddenUnits; i++)
			{
				result += hiddenResults[i] * this->HiddenToOutputWeights[startIndex + i];
			}

			outputResults[outputNeuron] = result;
		}
	}
};
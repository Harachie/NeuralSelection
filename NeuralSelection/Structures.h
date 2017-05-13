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

struct StockDataExtraction
{
	StockData *BuyBar;
	vector<StockData> *UsedStockData;
	vector<float> *Predictors;

	StockDataExtraction(StockData *buyBar)
	{
		this->BuyBar = buyBar;
		this->UsedStockData = new vector<StockData>();
		this->Predictors = new vector<float>();
	}

	~StockDataExtraction()
	{
		delete this->UsedStockData;
		delete this->Predictors;

		this->BuyBar = NULL;
		this->UsedStockData = NULL;
		this->Predictors = NULL;
	}

};

struct StockDataExtractionVector
{
	vector<StockDataExtraction> *Extractions;

	StockDataExtractionVector()
	{
		this->Extractions = new vector<StockDataExtraction>();
	}

	~StockDataExtractionVector()
	{
		delete this->Extractions;

		this->Extractions = NULL;
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

	StockDataExtractionVector* ExtractSteps(size_t stepSize, size_t count)
	{
		size_t startIndex;
		StockDataExtractionVector *r = new StockDataExtractionVector();
		StockDataExtraction *e;
		float percent;

		startIndex = stepSize * count + 1;

		for (size_t i = startIndex; i < this->Data->size(); i++)
		{

			e = new StockDataExtraction(&this->Data->at(i));

			for (size_t stepIndex = i - startIndex; stepIndex < i; stepIndex += stepSize)
			{
				e->UsedStockData->push_back(this->Data->at(stepIndex));
			}

			for (size_t p = 0; p < count; p++)
			{
				percent = e->UsedStockData->at(p + 1).Close / e->UsedStockData->at(p).Close;
				e->Predictors->push_back(percent);
			}

			r->Extractions->push_back(*e);
		}

		return r;
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

	size_t GetTotalWeightsCount()
	{
		return this->Predictors * this->HiddenUnits + this->HiddenUnits * this->OutputUnits + this->HiddenUnits + this->OutputUnits;
	}

	void SetNetworkWeights(const float *weights)
	{
		size_t index;
		size_t max;

		max = this->Predictors * this->HiddenUnits;

		for (size_t i = 0; i < max; i++)
		{
			this->InputToHiddenWeights[i] = weights[index++];
		}

		max = this->HiddenUnits * this->OutputUnits;

		for (size_t i = 0; i < max; i++)
		{
			this->HiddenToOutputWeights[i] = weights[index++];
		}

		for (size_t i = 0; i < this->HiddenUnits; i++)
		{
			this->HiddenBiasWeights[i] = weights[index++];
		}

		for (size_t i = 0; i < max; i++)
		{
			this->OutputBiasWeights[i] = weights[index++];
		}
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

	void CalculateSigmoid(float *inputs, float *hiddenResults, float *outputResults)
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

			hiddenResults[hiddenNeuron] = sigmoid(result);
		}

		for (size_t outputNeuron = 0; outputNeuron < this->OutputUnits; outputNeuron++)
		{
			result = this->OutputBiasWeights[outputNeuron];
			startIndex = outputNeuron * this->HiddenUnits;

			for (size_t i = 0; i < this->HiddenUnits; i++)
			{
				result += hiddenResults[i] * this->HiddenToOutputWeights[startIndex + i];
			}

			outputResults[outputNeuron] = sigmoid(result);
		}
	}
};
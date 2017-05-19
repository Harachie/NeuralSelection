#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <assert.h>
#include "IO.h"
#include "Randomization.h"
#include <algorithm>

using namespace std;

void TestSoftmax()
{
	float *values = new float[7];
	float *result;
	float softmaxSum;

	values[0] = 1.0f;
	values[1] = 2.0f;
	values[2] = 3.0f;
	values[3] = 4.0f;
	values[4] = 1.0f;
	values[5] = 2.0f;
	values[6] = 3.0f;

	result = softmax(values, 7, 1);
	softmaxSum = sum(result, 7);

	assert(softmaxSum == 1.0f);

	delete[] values;
	delete[] result;
}

void TestNetwork1()
{
	SimpleNeuralNetwork network(1, 3, 2);

	float *inputs, *hiddenResults, *outputResults;

	inputs = new float[1];
	inputs[0] = 1.0f;

	hiddenResults = network.CreateHiddenResultSet();
	outputResults = network.CreateOutputResultSet();

	network.InputToHiddenWeights[0] = 1.0f;
	network.InputToHiddenWeights[1] = 2.0f;
	network.InputToHiddenWeights[2] = 3.0f;

	network.HiddenToOutputWeights[0] = 1.0f;
	network.HiddenToOutputWeights[1] = 2.0f;
	network.HiddenToOutputWeights[2] = 3.0f;
	network.HiddenToOutputWeights[3] = 4.0f;
	network.HiddenToOutputWeights[4] = 5.0f;
	network.HiddenToOutputWeights[5] = 6.0f;

	network.HiddenBiasWeights[0] = 0.0f;
	network.HiddenBiasWeights[1] = 0.0f;
	network.HiddenBiasWeights[2] = 0.0f;

	network.OutputBiasWeights[0] = 0.0f;
	network.OutputBiasWeights[1] = 0.0f;

	network.CalculateRaw(inputs, hiddenResults, outputResults);

	assert(hiddenResults[0] == 1.0f);
	assert(hiddenResults[1] == 2.0f);
	assert(hiddenResults[2] == 3.0f);

	assert(outputResults[0] == 14.0f);
	assert(outputResults[1] == 32.0f);

	delete[] hiddenResults;
	delete[] outputResults;
}

void TestNetwork2()
{
	SimpleNeuralNetwork network(1, 3, 2);

	float *inputs, *hiddenResults, *outputResults;

	inputs = new float[1];
	inputs[0] = 1.0f;

	hiddenResults = network.CreateHiddenResultSet();
	outputResults = network.CreateOutputResultSet();

	network.InputToHiddenWeights[0] = 1.0f;
	network.InputToHiddenWeights[1] = 2.0f;
	network.InputToHiddenWeights[2] = 3.0f;

	network.HiddenToOutputWeights[0] = 1.0f;
	network.HiddenToOutputWeights[1] = 2.0f;
	network.HiddenToOutputWeights[2] = 3.0f;
	network.HiddenToOutputWeights[3] = 4.0f;
	network.HiddenToOutputWeights[4] = 5.0f;
	network.HiddenToOutputWeights[5] = 6.0f;

	network.HiddenBiasWeights[0] = 1.0f;
	network.HiddenBiasWeights[1] = 2.0f;
	network.HiddenBiasWeights[2] = 3.0f;

	network.OutputBiasWeights[0] = 1.0f;
	network.OutputBiasWeights[1] = 2.0f;

	network.CalculateRaw(inputs, hiddenResults, outputResults);

	assert(hiddenResults[0] == 2.0f);
	assert(hiddenResults[1] == 4.0f);
	assert(hiddenResults[2] == 6.0f);

	assert(outputResults[0] == 29.0f);
	assert(outputResults[1] == 66.0f);

	delete[] hiddenResults;
	delete[] outputResults;
}

void TestNetwork3()
{
	SimpleNeuralNetwork network(13, 3, 1);
	float *inputs, *hiddenResults, *outputResults;
}

unordered_set<uint32_t>* GetValidDates(vector<StockDataVector> &dataVectors)
{
	unordered_set<uint32_t>	*validDates = new unordered_set<uint32_t>();
	unordered_set<uint32_t> *dateSet;
	vector<unordered_set<uint32_t>> dateSets;
	size_t count = 0;
	size_t maxSetIndex;
	int canAdd;

	for (size_t i = 0; i < dataVectors.size(); i++)
	{
		dateSet = dataVectors.at(i).ExtractDates();
		dateSets.push_back(*dateSet);

		if (dateSet->size() > count)
		{
			count = dateSet->size();
			maxSetIndex = i;
		}
	}

	dateSet = &dateSets.at(maxSetIndex);

	for (auto element = dateSet->begin(); element != dateSet->end(); ++element)
	{
		canAdd = 1;

		for (size_t i = 0; i < dataVectors.size(); i++)
		{
			if (i != maxSetIndex) //sollte nicht das set sein, welches wir eh iterieren
			{
				if (!dateSets.at(i).count(*element))
				{
					canAdd = 0;

					break;
				}
			}
		}

		if (canAdd)
		{
			validDates->insert(*element);
		}
	}

	return validDates;
}

void Cars(string dataDirectory)
{
	size_t predictorsCount = 2;
	size_t stepSize = 65;
	uint32_t startDate = 20100101;

	vector<StockDataVector> dataVectors;
	vector<StockDataExtractionVector> extractionVectors;

	StockDataVector *vow, *dai, *bmw;
	StockDataVector *vowFiltered, *daiFiltered, *bmwFiltered;
	StockDataExtractionVector *vowSteps, *daiSteps, *bmwSteps;
	unordered_set<uint32_t> *validDates;
	SimpleNeuralNetwork network(predictorsCount, 5, 1);
	float *inputs, *hiddenResults, *outputResults, *randoms, *predictors;
	float *results, *softmaxResults;
	size_t dataCount, index, outputSetsCount;
	Xor1024 xor;

	initializeXor1024(xor);
	hiddenResults = network.CreateHiddenResultSet();
	outputResults = network.CreateOutputResultSet();
	randoms = new float[network.GetTotalWeightsCount()];
	generateRandoms(xor, randoms, network.GetTotalWeightsCount(), -5.0f, 5.0f);
	network.SetNetworkWeights(randoms);

	validDates = new unordered_set<uint32_t>();
	vow = ReadStockFile(dataDirectory + string("vow3.de.txt"));
	dai = ReadStockFile(dataDirectory + string("dai.de.txt"));
	bmw = ReadStockFile(dataDirectory + string("bmw.de.txt"));

	dataVectors.push_back(*vow);
	dataVectors.push_back(*dai);
	dataVectors.push_back(*bmw);

	validDates = GetValidDates(dataVectors);

	vowFiltered = vow->FilterByDate(validDates, startDate);
	daiFiltered = dai->FilterByDate(validDates, startDate);
	bmwFiltered = bmw->FilterByDate(validDates, startDate);

	vowSteps = vowFiltered->ExtractSteps(stepSize, predictorsCount);
	daiSteps = daiFiltered->ExtractSteps(stepSize, predictorsCount);
	bmwSteps = bmwFiltered->ExtractSteps(stepSize, predictorsCount);

	extractionVectors.push_back(*vowSteps);
	extractionVectors.push_back(*daiSteps);
	extractionVectors.push_back(*bmwSteps);

	dataCount = vowSteps->Extractions.size();
	outputSetsCount = dataCount * extractionVectors.size();
	results = new float[outputSetsCount];
	softmaxResults = new float[outputSetsCount];


	for (size_t i = 0, index = 0; i < outputSetsCount; i += 3, index++)
	{
		for (size_t n = 0; n < extractionVectors.size(); n++)
		{
			predictors = &extractionVectors.at(n).Extractions.at(index).Predictors.at(0);
			network.CalculateSigmoid(predictors, hiddenResults, outputResults);
			results[i + n] = outputResults[0];
		}
	}

	softmax(softmaxResults, results, extractionVectors.size(), dataCount);

	Depot test(extractionVectors.size());

	test.BuyEveryBar(dataCount, softmaxResults, extractionVectors, 100);
}



int main(int argc, char* argv[]) {
	string executablePath = argv[0];
	string directory = executablePath.substr(0, executablePath.length() - 19); //-neuralselection.exe
	string dataDirectory = directory + string("Data\\");
	vector<string> stockDataFiles = { "ads.de.txt", "alv.de.txt", "bas.de.txt", "bayn.de.txt", "bei.de.txt", "bmw.de.txt", "cbk.de.txt", "dai.de.txt", "dbk.de.txt", "dpw.de.txt", "dte.de.txt", "eoan.de.txt", "fme.de.txt", "fre.de.txt", "hei.de.txt", "hen3.de.txt", "ifx.de.txt", "lha.de.txt", "lin.de.txt", "mrk.de.txt", "muv2.de.txt", "psm.de.txt", "rwe.de.txt", "sap.de.txt", "sie.de.txt", "tka.de.txt", "vow3.de.txt", "_con.de.txt" };

	TestSoftmax();
	TestNetwork1();
	TestNetwork2();
	Cars(dataDirectory);

	return 0;
}
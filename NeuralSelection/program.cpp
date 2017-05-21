#pragma once

#include <string>
#include <inttypes.h>
#include <stdio.h>
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

//void TestNetwork3()
//{
//	SimpleNeuralNetwork network(13, 3, 1);
//	float *inputs, *hiddenResults, *outputResults;
//}

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
	size_t networkCount = 40;
	size_t predictorsCount = 4;
	size_t hiddenCount = 3;
	size_t stepSize = 65;
	uint32_t startDate = 19900101;

	vector<StockDataVector> dataVectors;
	vector<StockDataExtractionVector> extractionVectors;
	vector<SimpleNeuralNetwork> networks;

	/* die outputs durch ein andres netzwerk jagen und N outputs berechnen (also 3 rein N=3 outputs raus), darauf softmax*/


	for (size_t i = 0; i < networkCount; i++)
	{
		networks.push_back(SimpleNeuralNetwork(predictorsCount, hiddenCount, 1));
	}

	StockDataVector *vow, *dai, *bmw;
	StockDataVector *vowFiltered, *daiFiltered, *bmwFiltered;
	StockDataExtractionVector *vowSteps, *daiSteps, *bmwSteps;
	unordered_set<uint32_t> *validDates;

	float *inputs, *hiddenResults, *outputResults, *randoms, *predictors;
	float *results, *softmaxResults;
	size_t dataCount, index, outputSetsCount, totalWeightsCount;
	uint64_t *randomIndex = new uint64_t[1024];
	float *randomCrs, *adjustedWeights;
	float bestFitness = 0.0f;
	float compareEvenly;
	Xor1024 xor;

	initializeXor1024(xor);


	hiddenResults = networks.at(0).CreateHiddenResultSet();
	outputResults = networks.at(0).CreateOutputResultSet();
	totalWeightsCount = networks.at(0).TotalWeightsCount;
	randoms = new float[totalWeightsCount];
	randomCrs = new float[totalWeightsCount];
	adjustedWeights = new float[totalWeightsCount];
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

	Depot test(extractionVectors.size());
	SimpleNeuralNetwork *currentNetwork;
	SimpleNeuralNetwork testNetwork(predictorsCount, hiddenCount, 1);
	uint64_t aIndex, bIndex, cIndex;

	for (size_t i = 0; i < networkCount; i++)
	{
		generateRandoms(xor, randoms, totalWeightsCount, -5.0f, 5.0f);
		networks.at(i).SetNetworkWeights(randoms);
	}



	for (size_t networkIndex = 0; networkIndex < networkCount; networkIndex++)
	{
		currentNetwork = &networks.at(networkIndex);

		for (size_t i = 0, index = 0; i < outputSetsCount; i += 3, index++)
		{
			for (size_t n = 0; n < extractionVectors.size(); n++)
			{
				predictors = &extractionVectors.at(n).Extractions.at(index).Predictors.at(0);
				currentNetwork->CalculateSigmoidRawOutput(predictors, hiddenResults, outputResults);
				results[i + n] = outputResults[0];
			}
		}

		softmax(softmaxResults, results, extractionVectors.size(), dataCount);
		test.BuyEveryBar(dataCount, softmaxResults, extractionVectors, 100);
		currentNetwork->CurrentFitness = test.CurrentInvestmentValue;

		if (currentNetwork->CurrentFitness > bestFitness)
		{
			bestFitness = currentNetwork->CurrentFitness;
			printf("new best: %f\n", bestFitness);
		}
	}

	test.BuyEveryBarEvenly(dataCount, extractionVectors, 100);
	compareEvenly = test.CurrentInvestmentValue;
	uint64_t round = 0;

	do
	{
		round++;

		for (size_t networkIndex = 0; networkIndex < networkCount; networkIndex++)
		{
			generateRandoms(xor, randomIndex, 1024, networkCount);
			generateRandoms(xor, randomCrs, totalWeightsCount);

			for (size_t i = 0; i < 1021; i++)
			{
				if ((randomIndex[i] != networkIndex) && (randomIndex[i + 1] != networkIndex) && (randomIndex[i + 2] != networkIndex))
				{
					if ((randomIndex[i] != randomIndex[i + 1]) && (randomIndex[i] != randomIndex[i + 2]) && (randomIndex[i + 1] != randomIndex[i + 2]))
					{
						aIndex = randomIndex[i];
						bIndex = randomIndex[i + 1];
						cIndex = randomIndex[i + 2];

						break;
					}
				}
			}

			for (size_t i = 0; i < totalWeightsCount; i++)
			{
				if (randomCrs[i] < 0.9f)
				{
					adjustedWeights[i] = networks.at(aIndex).Weights[i] + 0.8f * (networks.at(bIndex).Weights[i] - networks.at(cIndex).Weights[i]);
				}
				else
				{
					adjustedWeights[i] = networks.at(networkIndex).Weights[i];
				}
			}

			testNetwork.SetNetworkWeights(adjustedWeights);

			for (size_t i = 0, index = 0; i < outputSetsCount; i += 3, index++)
			{
				for (size_t n = 0; n < extractionVectors.size(); n++)
				{
					predictors = &extractionVectors.at(n).Extractions.at(index).Predictors.at(0);
					testNetwork.CalculateSigmoidRawOutput(predictors, hiddenResults, outputResults);
					results[i + n] = outputResults[0];
				}
			}

			softmax(softmaxResults, results, extractionVectors.size(), dataCount);
			test.BuyEveryBar(dataCount, softmaxResults, extractionVectors, 100);

			if (test.CurrentInvestmentValue > networks.at(networkIndex).CurrentFitness)
			{
				networks.at(networkIndex).CurrentFitness = test.CurrentInvestmentValue;
				networks.at(networkIndex).SetNetworkWeights(testNetwork.Weights);

				if (test.CurrentInvestmentValue > bestFitness)
				{
					bestFitness = test.CurrentInvestmentValue;
					printf("new best: %f (%.2f) at %" PRIu64 " \n", bestFitness, ((bestFitness / compareEvenly) - 1.0f) * 100.0f, round);
				}
			}
		}

	} while (true);


}


void CarsWithTest(string dataDirectory)
{
	size_t networkCount = 40;
	size_t predictorsCount = 4;
	size_t hiddenCount = 3;
	size_t stepSize = 65;
	uint32_t startDate = 19900101;

	vector<StockDataVector> dataVectors;
	vector<StockDataExtractionVector> extractionVectors;
	vector<StockDataExtractionVector> extractionVectorsTest;
	vector<SimpleNeuralNetwork> networks;

	/* die outputs durch ein andres netzwerk jagen und N outputs berechnen (also 3 rein N=3 outputs raus), darauf softmax*/


	for (size_t i = 0; i < networkCount; i++)
	{
		networks.push_back(SimpleNeuralNetwork(predictorsCount, hiddenCount, 1));
	}

	StockDataVector *vow, *dai, *bmw, *alv, *sie, *bayn;
	StockDataVector *vowFiltered, *daiFiltered, *bmwFiltered, *alvF, *sieF, *baynF;
	StockDataExtractionVector *vowSteps, *daiSteps, *bmwSteps, *alvS, *sieS, *baynS;
	unordered_set<uint32_t> *validDates;

	float *inputs, *hiddenResults, *outputResults, *randoms, *predictors;
	float *results, *softmaxResults;
	size_t dataCount, index, outputSetsCount, totalWeightsCount;
	uint64_t *randomIndex = new uint64_t[1024];
	float *randomCrs, *adjustedWeights;
	float bestFitness = 0.0f;
	float compareEvenly, compareEvenlyTest;
	Xor1024 xor;

	initializeXor1024(xor);


	hiddenResults = networks.at(0).CreateHiddenResultSet();
	outputResults = networks.at(0).CreateOutputResultSet();
	totalWeightsCount = networks.at(0).TotalWeightsCount;
	randoms = new float[totalWeightsCount];
	randomCrs = new float[totalWeightsCount];
	adjustedWeights = new float[totalWeightsCount];
	validDates = new unordered_set<uint32_t>();

	vow = ReadStockFile(dataDirectory + string("vow3.de.txt"));
	dai = ReadStockFile(dataDirectory + string("dai.de.txt"));
	bmw = ReadStockFile(dataDirectory + string("bmw.de.txt"));

	alv = ReadStockFile(dataDirectory + string("alv.de.txt"));
	sie = ReadStockFile(dataDirectory + string("sie.de.txt"));
	bayn = ReadStockFile(dataDirectory + string("bayn.de.txt"));

	dataVectors.push_back(*vow);
	dataVectors.push_back(*dai);
	dataVectors.push_back(*bmw);
	dataVectors.push_back(*alv);
	dataVectors.push_back(*sie);
	dataVectors.push_back(*bayn);

	validDates = GetValidDates(dataVectors);

	vowFiltered = vow->FilterByDate(validDates, startDate);
	daiFiltered = dai->FilterByDate(validDates, startDate);
	bmwFiltered = bmw->FilterByDate(validDates, startDate);

	alvF = alv->FilterByDate(validDates, startDate);
	sieF = sie->FilterByDate(validDates, startDate);
	baynF = bayn->FilterByDate(validDates, startDate);

	vowSteps = vowFiltered->ExtractSteps(stepSize, predictorsCount);
	daiSteps = daiFiltered->ExtractSteps(stepSize, predictorsCount);
	bmwSteps = bmwFiltered->ExtractSteps(stepSize, predictorsCount);

	alvS = alvF->ExtractSteps(stepSize, predictorsCount);
	sieS = sieF->ExtractSteps(stepSize, predictorsCount);
	baynS = baynF->ExtractSteps(stepSize, predictorsCount);

	extractionVectors.push_back(*vowSteps);
	extractionVectors.push_back(*daiSteps);
	extractionVectors.push_back(*bmwSteps);

	extractionVectorsTest.push_back(*alvS);
	extractionVectorsTest.push_back(*sieS);
	extractionVectorsTest.push_back(*baynS);

	dataCount = vowSteps->Extractions.size();
	outputSetsCount = dataCount * extractionVectors.size();
	results = new float[outputSetsCount];
	softmaxResults = new float[outputSetsCount];

	Depot test(extractionVectors.size());
	SimpleNeuralNetwork *currentNetwork;
	SimpleNeuralNetwork testNetwork(predictorsCount, hiddenCount, 1);
	uint64_t aIndex, bIndex, cIndex;

	for (size_t i = 0; i < networkCount; i++)
	{
		generateRandoms(xor, randoms, totalWeightsCount, -5.0f, 5.0f);
		networks.at(i).SetNetworkWeights(randoms);
	}



	for (size_t networkIndex = 0; networkIndex < networkCount; networkIndex++)
	{
		currentNetwork = &networks.at(networkIndex);

		for (size_t i = 0, index = 0; i < outputSetsCount; i += 3, index++)
		{
			for (size_t n = 0; n < extractionVectors.size(); n++)
			{
				predictors = &extractionVectors.at(n).Extractions.at(index).Predictors.at(0);
				currentNetwork->CalculateSigmoidRawOutput(predictors, hiddenResults, outputResults);
				results[i + n] = outputResults[0];
			}
		}

		softmax(softmaxResults, results, extractionVectors.size(), dataCount);
		test.BuyEveryBar(dataCount, softmaxResults, extractionVectors, 100);
		currentNetwork->CurrentFitness = test.CurrentInvestmentValue;

		if (currentNetwork->CurrentFitness > bestFitness)
		{
			bestFitness = currentNetwork->CurrentFitness;
			printf("new best: %f\n", bestFitness);
		}
	}

	test.BuyEveryBarEvenly(dataCount, extractionVectors, 100);
	compareEvenly = test.CurrentInvestmentValue;

	test.BuyEveryBarEvenly(dataCount, extractionVectorsTest, 100);
	compareEvenlyTest = test.CurrentInvestmentValue;
	uint64_t round = 0;

	do
	{
		round++;

		for (size_t networkIndex = 0; networkIndex < networkCount; networkIndex++)
		{
			generateRandoms(xor, randomIndex, 1024, networkCount);
			generateRandoms(xor, randomCrs, totalWeightsCount);

			for (size_t i = 0; i < 1021; i++)
			{
				if ((randomIndex[i] != networkIndex) && (randomIndex[i + 1] != networkIndex) && (randomIndex[i + 2] != networkIndex))
				{
					if ((randomIndex[i] != randomIndex[i + 1]) && (randomIndex[i] != randomIndex[i + 2]) && (randomIndex[i + 1] != randomIndex[i + 2]))
					{
						aIndex = randomIndex[i];
						bIndex = randomIndex[i + 1];
						cIndex = randomIndex[i + 2];

						break;
					}
				}
			}

			for (size_t i = 0; i < totalWeightsCount; i++)
			{
				if (randomCrs[i] < 0.9f)
				{
					adjustedWeights[i] = networks.at(aIndex).Weights[i] + 0.8f * (networks.at(bIndex).Weights[i] - networks.at(cIndex).Weights[i]);
				}
				else
				{
					adjustedWeights[i] = networks.at(networkIndex).Weights[i];
				}
			}

			testNetwork.SetNetworkWeights(adjustedWeights);

			for (size_t i = 0, index = 0; i < outputSetsCount; i += 3, index++)
			{
				for (size_t n = 0; n < extractionVectors.size(); n++)
				{
					predictors = &extractionVectors.at(n).Extractions.at(index).Predictors.at(0);
					testNetwork.CalculateSigmoidRawOutput(predictors, hiddenResults, outputResults);
					results[i + n] = outputResults[0];
				}
			}

			softmax(softmaxResults, results, extractionVectors.size(), dataCount);
			test.BuyEveryBar(dataCount, softmaxResults, extractionVectors, 100);

			if (test.CurrentInvestmentValue > networks.at(networkIndex).CurrentFitness)
			{
				networks.at(networkIndex).CurrentFitness = test.CurrentInvestmentValue;
				networks.at(networkIndex).SetNetworkWeights(testNetwork.Weights);

				if (test.CurrentInvestmentValue > bestFitness)
				{
					bestFitness = test.CurrentInvestmentValue;
					printf("new best: %f (%.2f) at %" PRIu64, bestFitness, ((bestFitness / compareEvenly) - 1.0f) * 100.0f, round);


					for (size_t i = 0, index = 0; i < outputSetsCount; i += 3, index++)
					{
						for (size_t n = 0; n < extractionVectorsTest.size(); n++)
						{
							predictors = &extractionVectorsTest.at(n).Extractions.at(index).Predictors.at(0);
							testNetwork.CalculateSigmoidRawOutput(predictors, hiddenResults, outputResults);
							results[i + n] = outputResults[0];
						}
					}

					softmax(softmaxResults, results, extractionVectorsTest.size(), dataCount);
					test.BuyEveryBar(dataCount, softmaxResults, extractionVectorsTest, 100);
					printf(", compare: %f (%.2f) at %" PRIu64 " \n", test.CurrentInvestmentValue, ((test.CurrentInvestmentValue / compareEvenlyTest) - 1.0f) * 100.0f, round);

				}
			}
		}

	} while (true);


}


void Stocks6(string dataDirectory)
{
	size_t networkCount = 40;
	size_t predictorsCount = 6;
	size_t hiddenCount = 3;
	size_t stepSize = 65;
	uint32_t startDate = 19900101;

	vector<StockDataVector> dataVectors;
	vector<StockDataExtractionVector> extractionVectors;
	vector<StockDataExtractionVector> extractionVectorsTest;
	vector<SimpleNeuralNetwork> networks;

	/* die outputs durch ein andres netzwerk jagen und N outputs berechnen (also 3 rein N=3 outputs raus), darauf softmax*/


	for (size_t i = 0; i < networkCount; i++)
	{
		networks.push_back(SimpleNeuralNetwork(predictorsCount, hiddenCount, 1));
	}

	StockDataVector *vow, *dai, *bmw, *alv, *sie, *bayn;
	StockDataVector *vowFiltered, *daiFiltered, *bmwFiltered, *alvF, *sieF, *baynF;
	StockDataExtractionVector *vowSteps, *daiSteps, *bmwSteps, *alvS, *sieS, *baynS;
	unordered_set<uint32_t> *validDates;

	float *inputs, *hiddenResults, *outputResults, *randoms, *predictors;
	float *results, *softmaxResults;
	size_t dataCount, index, outputSetsCount, totalWeightsCount;
	uint64_t *randomIndex = new uint64_t[1024];
	float *randomCrs, *adjustedWeights;
	float bestFitness = 0.0f;
	float compareEvenly;
	Xor1024 xor;

	initializeXor1024(xor);


	hiddenResults = networks.at(0).CreateHiddenResultSet();
	outputResults = networks.at(0).CreateOutputResultSet();
	totalWeightsCount = networks.at(0).TotalWeightsCount;
	randoms = new float[totalWeightsCount];
	randomCrs = new float[totalWeightsCount];
	adjustedWeights = new float[totalWeightsCount];
	validDates = new unordered_set<uint32_t>();

	vow = ReadStockFile(dataDirectory + string("vow3.de.txt"));
	dai = ReadStockFile(dataDirectory + string("dai.de.txt"));
	bmw = ReadStockFile(dataDirectory + string("bmw.de.txt"));

	alv = ReadStockFile(dataDirectory + string("alv.de.txt"));
	sie = ReadStockFile(dataDirectory + string("sie.de.txt"));
	bayn = ReadStockFile(dataDirectory + string("bayn.de.txt"));

	dataVectors.push_back(*vow);
	dataVectors.push_back(*dai);
	dataVectors.push_back(*bmw);
	dataVectors.push_back(*alv);
	dataVectors.push_back(*sie);
	dataVectors.push_back(*bayn);

	validDates = GetValidDates(dataVectors);

	vowFiltered = vow->FilterByDate(validDates, startDate);
	daiFiltered = dai->FilterByDate(validDates, startDate);
	bmwFiltered = bmw->FilterByDate(validDates, startDate);

	alvF = alv->FilterByDate(validDates, startDate);
	sieF = sie->FilterByDate(validDates, startDate);
	baynF = bayn->FilterByDate(validDates, startDate);

	vowSteps = vowFiltered->ExtractSteps(stepSize, predictorsCount);
	daiSteps = daiFiltered->ExtractSteps(stepSize, predictorsCount);
	bmwSteps = bmwFiltered->ExtractSteps(stepSize, predictorsCount);

	alvS = alvF->ExtractSteps(stepSize, predictorsCount);
	sieS = sieF->ExtractSteps(stepSize, predictorsCount);
	baynS = baynF->ExtractSteps(stepSize, predictorsCount);

	extractionVectors.push_back(*vowSteps);
	extractionVectors.push_back(*daiSteps);
	extractionVectors.push_back(*bmwSteps);

	extractionVectors.push_back(*alvS);
	extractionVectors.push_back(*sieS);
	extractionVectors.push_back(*baynS);

	dataCount = vowSteps->Extractions.size();
	outputSetsCount = dataCount * extractionVectors.size();
	results = new float[outputSetsCount];
	softmaxResults = new float[outputSetsCount];

	Depot test(extractionVectors.size());
	SimpleNeuralNetwork *currentNetwork;
	SimpleNeuralNetwork testNetwork(predictorsCount, hiddenCount, 1);
	uint64_t aIndex, bIndex, cIndex;

	for (size_t i = 0; i < networkCount; i++)
	{
		generateRandoms(xor, randoms, totalWeightsCount, -5.0f, 5.0f);
		networks.at(i).SetNetworkWeights(randoms);
	}



	for (size_t networkIndex = 0; networkIndex < networkCount; networkIndex++)
	{
		currentNetwork = &networks.at(networkIndex);

		for (size_t i = 0, index = 0; i < outputSetsCount; i += extractionVectors.size(), index++)
		{
			for (size_t n = 0; n < extractionVectors.size(); n++)
			{
				predictors = &extractionVectors.at(n).Extractions.at(index).Predictors.at(0);
				currentNetwork->CalculateSigmoidRawOutput(predictors, hiddenResults, outputResults);
				results[i + n] = outputResults[0];
			}
		}

		softmax(softmaxResults, results, extractionVectors.size(), dataCount);
		test.BuyEveryBar(dataCount, softmaxResults, extractionVectors, 100);
		currentNetwork->CurrentFitness = test.CurrentInvestmentValue;

		if (currentNetwork->CurrentFitness > bestFitness)
		{
			bestFitness = currentNetwork->CurrentFitness;
			printf("new best: %f\n", bestFitness);
		}
	}

	test.BuyEveryBarEvenly(dataCount, extractionVectors, 100);
	compareEvenly = test.CurrentInvestmentValue;
	uint64_t round = 0;

	do
	{
		round++;

		for (size_t networkIndex = 0; networkIndex < networkCount; networkIndex++)
		{
			generateRandoms(xor, randomIndex, 1024, networkCount);
			generateRandoms(xor, randomCrs, totalWeightsCount);

			for (size_t i = 0; i < 1021; i++)
			{
				if ((randomIndex[i] != networkIndex) && (randomIndex[i + 1] != networkIndex) && (randomIndex[i + 2] != networkIndex))
				{
					if ((randomIndex[i] != randomIndex[i + 1]) && (randomIndex[i] != randomIndex[i + 2]) && (randomIndex[i + 1] != randomIndex[i + 2]))
					{
						aIndex = randomIndex[i];
						bIndex = randomIndex[i + 1];
						cIndex = randomIndex[i + 2];

						break;
					}
				}
			}

			for (size_t i = 0; i < totalWeightsCount; i++)
			{
				if (randomCrs[i] < 0.9f)
				{
					adjustedWeights[i] = networks.at(aIndex).Weights[i] + 0.8f * (networks.at(bIndex).Weights[i] - networks.at(cIndex).Weights[i]);
				}
				else
				{
					adjustedWeights[i] = networks.at(networkIndex).Weights[i];
				}
			}

			testNetwork.SetNetworkWeights(adjustedWeights);

			for (size_t i = 0, index = 0; i < outputSetsCount; i += extractionVectors.size(), index++)
			{
				for (size_t n = 0; n < extractionVectors.size(); n++)
				{
					predictors = &extractionVectors.at(n).Extractions.at(index).Predictors.at(0);
					testNetwork.CalculateSigmoidRawOutput(predictors, hiddenResults, outputResults);
					results[i + n] = outputResults[0];
				}
			}

			softmax(softmaxResults, results, extractionVectors.size(), dataCount);
			test.BuyEveryBar(dataCount, softmaxResults, extractionVectors, 100);

			if (test.CurrentInvestmentValue > networks.at(networkIndex).CurrentFitness)
			{
				networks.at(networkIndex).CurrentFitness = test.CurrentInvestmentValue;
				networks.at(networkIndex).SetNetworkWeights(testNetwork.Weights);

				if (test.CurrentInvestmentValue > bestFitness)
				{
					bestFitness = test.CurrentInvestmentValue;
					printf("new best: %f (%.2f) at %" PRIu64 "\n", bestFitness, ((bestFitness / compareEvenly) - 1.0f) * 100.0f, round);


				}
			}
		}

	} while (true);


}

int main(int argc, char* argv[]) {
	string executablePath = argv[0];
	string directory = executablePath.substr(0, executablePath.length() - 19); //-neuralselection.exe
	string dataDirectory = directory + string("Data\\");
	vector<string> stockDataFiles = { "ads.de.txt", "alv.de.txt", "bas.de.txt", "bayn.de.txt", "bei.de.txt", "bmw.de.txt", "cbk.de.txt", "dai.de.txt", "dbk.de.txt", "dpw.de.txt", "dte.de.txt", "eoan.de.txt", "fme.de.txt", "fre.de.txt", "hei.de.txt", "hen3.de.txt", "ifx.de.txt", "lha.de.txt", "lin.de.txt", "mrk.de.txt", "muv2.de.txt", "psm.de.txt", "rwe.de.txt", "sap.de.txt", "sie.de.txt", "tka.de.txt", "vow3.de.txt", "_con.de.txt" };

	TestSoftmax();
	TestNetwork1();
	TestNetwork2();
	Stocks6(dataDirectory);

	return 0;
}
#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <assert.h>
#include "IO.h"
#include "Randomization.h"

using namespace std;

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

void Cars(string dataDirectory)
{
	StockDataVector *vow, *dai, *bmw;
	StockDataVector *vowFiltered, *daiFiltered, *bmwFiltered;
	StockDataExtractionVector *vowSteps, *daiSteps, *bmwSteps;
	unordered_set<uint32_t> *vowDates, *daiDates, *bmwDates, *validDates;
	SimpleNeuralNetwork network(13, 3, 1);
	float *inputs, *hiddenResults, *outputResults, *randoms, *predictors;
	float *results;
	Xor1024 xor;

	initializeXor1024(xor);

	hiddenResults = network.CreateHiddenResultSet();
	outputResults = network.CreateOutputResultSet();
	randoms = new float[network.GetTotalWeightsCount()];
	generateRandoms(xor, randoms, network.GetTotalWeightsCount(), -1.0f, 1.0f);
	network.SetNetworkWeights(randoms);

	validDates = new unordered_set<uint32_t>();
	vow = ReadStockFile(dataDirectory + string("vow3.de.txt"));
	dai = ReadStockFile(dataDirectory + string("dai.de.txt"));
	bmw = ReadStockFile(dataDirectory + string("bmw.de.txt"));

	vowDates = vow->ExtractDates();
	daiDates = dai->ExtractDates();
	bmwDates = bmw->ExtractDates();

	for (auto element = vowDates->begin(); element != vowDates->end(); ++element)
	{
		if (daiDates->count(*element) && bmwDates->count(*element))
		{
			validDates->insert(*element);
		}
	}

	vowFiltered = vow->FilterByDate(validDates, 20000101);
	daiFiltered = dai->FilterByDate(validDates, 20000101);
	bmwFiltered = bmw->FilterByDate(validDates, 20000101);

	vowSteps = vowFiltered->ExtractSteps(5, 13);
	daiSteps = daiFiltered->ExtractSteps(5, 13);
	bmwSteps = bmwFiltered->ExtractSteps(5, 13);

	results = new float[vowSteps->Extractions->size() * 3];

	//todo calc all outputs for 3 car sets etc
	for (size_t i = 0; i < vowSteps->Extractions->size(); i++)
	{
		predictors = &vowSteps->Extractions->at(i).Predictors->at(0);
		network.CalculateSigmoid(predictors, hiddenResults, outputResults);
		results[i] = outputResults[0];
	}

}

int main(int argc, char* argv[]) {
	string executablePath = argv[0];
	string directory = executablePath.substr(0, executablePath.length() - 19); //-neuralselection.exe
	string dataDirectory = directory + string("Data\\");
	vector<string> stockDataFiles = { "ads.de.txt", "alv.de.txt", "bas.de.txt", "bayn.de.txt", "bei.de.txt", "bmw.de.txt", "cbk.de.txt", "dai.de.txt", "dbk.de.txt", "dpw.de.txt", "dte.de.txt", "eoan.de.txt", "fme.de.txt", "fre.de.txt", "hei.de.txt", "hen3.de.txt", "ifx.de.txt", "lha.de.txt", "lin.de.txt", "mrk.de.txt", "muv2.de.txt", "psm.de.txt", "rwe.de.txt", "sap.de.txt", "sie.de.txt", "tka.de.txt", "vow3.de.txt", "_con.de.txt" };


	TestNetwork1();
	TestNetwork2();
	Cars(dataDirectory);



	StockDataVector *v;
	v = ReadStockFile(dataDirectory + stockDataFiles[0]);
	auto k = v->ExtractSteps(5, 13);
	return 0;
}
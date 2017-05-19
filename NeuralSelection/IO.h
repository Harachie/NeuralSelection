#pragma once

#include "stdio.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Structures.h"

using namespace std;

void Remove(string *source, const string &stringToRemove)
{
	size_t index = 0;


	while (true)
	{
		index = source->find(stringToRemove, index);

		if (index == string::npos)
		{
			break;
		}

		source->replace(index, stringToRemove.length(), "");
	}
}

static StockDataVector* ReadFile(const char *path)
{
	StockDataVector *r = new StockDataVector();
	errno_t err;
	char temp[4096];
	FILE *fp;
	int c = 0;
	size_t
		kpo;
	float
		ft = 0.0f;
	StockData *sd;
	string t;

	err = fopen_s(&fp, path, "r");
	sd = new StockData();
	sd->Volume = 0;

	if (err != 0)
	{
		printf("Could not open file.\n");
	}
	else if (fp != NULL)
	{
		while (fgets(temp, 4096, fp) != NULL)
		{
			t = string(temp);
			Remove(&t, string("-"));
			kpo = sscanf_s(t.c_str(), "%lu,%f,%f,%f,%f", &sd->Date, &sd->Open, &sd->High, &sd->Low, &sd->Close); //lu

			if (kpo > 0)
			{
				r->Data.push_back(StockData(sd->Date, sd->Open, sd->High, sd->Low, sd->Close, sd->Volume));
			}
		}

		fclose(fp);
	}

	delete sd;

	return r;
}

static StockDataVector* ReadStockFile(std::string fileName)
{
	StockDataVector *fileVector;

	fileVector = ReadFile((fileName).c_str());
	fileVector->Description = fileName;

	return fileVector;
}

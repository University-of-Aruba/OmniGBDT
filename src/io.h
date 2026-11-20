//
// Created by zhang on 19-7-4.
//

#ifndef MGBDT_IO_H
#define MGBDT_IO_H

#include "tree.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <limits.h>

using namespace std;

// functions for string processing
std::string lstrip(const std::string &str, const std::string &chars = "");

std::string rstrip(const std::string &str, const std::string &chars = "");

std::string strip(const std::string &str, const std::string &chars = "");

void split(const std::string &str, std::vector<std::string> &result, const std::string &sep = "", int maxsplit = -1);

std::string zfill(const std::string &str, int width);

//functions for dump and load the learned trees (to and from txt file)
int DumpTrees(vector<Tree> &, const vector<double> &, const char *);

int LoadTrees(vector<Tree> &, vector<double> &, const char *, int inp_dim, int out_dim, int round_width);

#endif //MGBDT_IO_H

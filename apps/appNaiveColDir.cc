/*
 * Copyright (C) 2023  Ferdous,S M <ferdous.csebuet@egmail.com>
 * Author: Ferdous,S M <ferdous.csebuet@egmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <cstring>
#include<fstream>
#include <random>

#include "ClqPart/JsonGraph.h"
#include "ClqPart/paletteCol.h"
#include "ClqPart/input.h"
#include "cxxopts/cxxopts.hpp"
#include "ClqPart/utility.h"
#include "ClqPart/greedyColor.h"
#include "ClqPart/MemUsage.h"


int main(int argC, char *argV[]) {
  
  cxxopts::Options options("naivecol", "read json pauli string files and color the graph using a naive greedy coloring algorithm. "); 
  options.add_options()
    ("in,infile", "input json file name", cxxopts::value<std::string>())
//("out,outfile", "output color file name", cxxopts::value<std::string>()->default_value(""))
    ("o,order", "NATURAL,RANDOM",cxxopts::value<std::string>()->default_value("RANDOM"))
    ("sd,seed", "use seed", cxxopts::value<int>()->default_value("123"))
    ("h,help", "print usage")
    ;

  auto baseline = getPeakRSS();
  //std::string inFname, outFname,order;
  std::string inFname,orderName;
  int seed;
  try{
    auto result = options.parse(argC,argV);
    if (result.count("help")) {
          std::cout<< options.help()<<"\n";
          std::exit(0);
    }
    inFname = result["infile"].as<std::string>();
    orderName = result["order"].as<std::string>();
    seed = result["seed"].as<int>();
  }
  catch(cxxopts::exceptions::exception &exp) {
    std::cout<<options.help()<<std::endl;
    exit(1);
  }

  ClqPart::JsonGraph jsongraph(inFname,false,true); 
  NODE_T n = jsongraph.numOfData();
  //jsongraph.printData();

  /*
  std::vector<NODE_T> forbiddenCol(n,-1);
  std::vector<NODE_T> colors(n,-1);
  
  colors[0] = 0;
  for(auto u=1 ; u<n; u++) {
    //populate the forbidden colors for u
    for(auto v=0; v<u; v++) {
      if(jsongraph.is_an_edge(v,u) == false) { 
        if (colors[v] != -1) {
          forbiddenCol[colors[v]] = u;
        }
      }
    } 
    //color u with first available color
    for( auto i=0;i<n;i++) {
      if(forbiddenCol[i] == -1) {
        colors[u] = i; 
        break;
      } 
    }
  }
  
  NODE_T nColors = *std::max_element(colors.begin(),colors.end()) + 1;
  */
  PaletteColor palcol(n);
  std::vector<NODE_T> vertList(n);
  std::iota(vertList.begin(),vertList.end(),0);

  if(orderName == "RANDOM") {
    std::cout<<"using random ordering, seed: "<<seed<<std::endl; 
    std::mt19937 engine(seed);
    std::shuffle(vertList.begin(),vertList.end(),engine);
  }
  else {
    std::cout<<"using natural ordering; ignoring seed" <<std::endl; 
  }
  

  palcol.naiveGreedyColor<std::vector<uint32_t>>(vertList,jsongraph,0);
  auto nColors = palcol.getNumColors();
  auto palstat = palcol.getPalStat();

  auto mem = getPeakRSS() - baseline;

  std::cout<<jsongraph.numOfData()<<","
    <<jsongraph.getNumEdge()<<","
    <<nColors<<","<<palstat.invColorTime<<","<<mem<<std::endl;
  return 0;
}  

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

void printStat( int level, PalColStat &palStat) {
  
  std::cout<<"***********Level "<<level<<"*******"<<std::endl;
  std::cout<<"Num Nodes: "<<palStat.n<<"\n";
  std::cout<<"Num Edges: "<<palStat.m<<"\n";
  std::cout<<"Avg. Deg.: "<<(double) 2*palStat.m/palStat.n<<"\n";
  std::cout<<"Palette Size: "<<palStat.palSz<<"\n";
  std::cout<<"List Size: "<<palStat.lstSz<<"\n";
  std::cout<<"Num Conflict Edges: "<<palStat.mConf<<"\n";
  std::cout<<"Conflict to Edge (%): "<<(double)palStat.mConf/palStat.m*100<<"\n";
  std::cout<<"Num Colors: " <<palStat.nColors<<std::endl;
  std::cout<<"Assign Time: "<<palStat.assignTime<<"\n";
  std::cout<<"Conf. Build Time: "<<palStat.confBuildTime<<"\n";
  std::cout<<"Conf. Color Time: "<<palStat.confColorTime<<"\n";
  std::cout<<"\n";

}

int main(int argC, char *argV[]) {
  
  cxxopts::Options options("palColGr", "read json pauli string files and color the graph using palette coloring algorithm (GPU-accelerated vesion)"); 
  options.add_options()
    ("in,infile", "json file containing the pauli strings", cxxopts::value<std::string>())
    ("out,outfile", "json file containing the groups after coloring", cxxopts::value<std::string>()->default_value(""))
    ("t,target", "palette size. Two choices: 1) absolute number (>=1), 2) percentage(0-1) of nodes", cxxopts::value<double>())
    ("a,alpha", "coefficient to log(n) for list size", cxxopts::value<float>()->default_value("1.0"))
    ("l,list", "use explicit list size", cxxopts::value<NODE_T>()->default_value("-1"))
    ("inv,ninv", "number of invalid vertices tolerance", cxxopts::value<NODE_T>()->default_value("100"))
    ("o,order", "RANDOM, LIST",cxxopts::value<std::string>()->default_value("LIST"))
    ("c,check", "check validity of coloring", cxxopts::value<bool>()->default_value("false"))
    ("r,recurse", "use recursive coloring", cxxopts::value<bool>()->default_value("false"))
    ("sd,seed", "use seed", cxxopts::value<int>()->default_value("123"))
    ("h,help", "print usage")
    ;

  std::string inFname,orderName,outFname;
  int seed;
  double target1;
  NODE_T target,list_size,nInv;
  float alpha;
  bool isValid,isRec;
  try{
    auto result = options.parse(argC,argV);
    if (result.count("help")) {
          std::cout<< options.help()<<"\n";
          std::exit(0);
    }
    inFname = result["infile"].as<std::string>();
    outFname = result["outfile"].as<std::string>();
    orderName = result["order"].as<std::string>();
    target1 = result["target"].as<double>();
    alpha = result["alpha"].as<float>();
    seed = result["seed"].as<int>();
    //isStream = result["stream"].as<bool>();
    isValid = result["check"].as<bool>();
    isRec = result["recurse"].as<bool>();
    list_size = result["list"].as<NODE_T>();
    nInv = result["ninv"].as<NODE_T>();
  }
  catch(cxxopts::exceptions::exception &exp) {
    std::cout<<options.help()<<std::endl;
    exit(1);
  }

  ClqPart::JsonGraph jsongraph(inFname, false, true); 
  NODE_T n = jsongraph.numOfData();
  
  double nextFrac;
  if(target1 < 1) {
    std::cout<<"Using target as node percentage"<<std::endl; 
    target = NODE_T(n*target1);
    nextFrac = target1;
  }
  else {
    target = NODE_T(target1); 
    nextFrac = 1.0/8.0;
  }
  if(list_size >=0)
    std::cout<<"Since list size is given, ignoring alpha"<<std::endl;
  bool Edge32Bit = n < (1 << 16);
  if(Edge32Bit){
    std::cout << "Using 32-bit offsets" << std::endl;
    PaletteColor<unsigned int> palcol(n,target,alpha,list_size,seed);
  
    int level = 0;
    palcol.buildConfGraphGpuMemConscious(jsongraph);
    if(orderName == "RANDOM"){
      palcol.confColorRandCSR();
    }
    else{
      palcol.confColorGreedyCSR();  
    }
    std::vector <NODE_T>  invVert = palcol.getInvVertices();
    PalColStat palStat = palcol.getPalStat(level); 
    palStat.m = jsongraph.getNumEdge();
    palStat.nColors = palcol.getNumColors(); 
    // std::cout<<"Invalid Vert: "<<invVert.size()<<"\n";
    printStat(level,palStat);
    if (isRec == true) {
        while(invVert.size() > nInv) { 
            jsongraph.resetNumEdge();
            level++;
            if(invVert.empty() == false) {
                // if(invVert.size() > 40000) alpha = 3;
                // else if (invVert.size() > 20000) alpha = 2; 
                // else if (invVert.size() > 5000) alpha = 1.5; 
                // else alpha = 1;
                // std::cout <<nextFrac<<std::endl;
                palcol.reInit(invVert,invVert.size()*nextFrac,alpha);
                palcol.buildConfGraphGpuMemConscious(jsongraph,invVert);
                // std::cout << "Greedy Coloring on GPU" << std::endl;
                if(orderName == "RANDOM"){
                  palcol.confColorRandCSR(invVert);
                }
                else{
                  palcol.confColorGreedyCSR(invVert);
                }
                // std::cout << "Greedy Coloring on GPU Done" << std::endl;
            }
            invVert = palcol.getInvVertices();
            palStat = palcol.getPalStat(level); 
            palStat.m = jsongraph.getNumEdge();
            //palStat.mConf = palcol.getNumConflicts();
            palStat.nColors = palcol.getNumColors(); 
            // std::cout<<"Invalid Vert: "<<invVert.size()<<"\n";
            printStat(level,palStat);
        }
    }
    if(invVert.empty() == false) {
        std::cout<<"Final Num invalid Vert: "<<invVert.size()<<"\n";
        palcol.naiveGreedyColor<std::vector<uint32_t>>(invVert, jsongraph, palcol.getNumColors());
        palStat = palcol.getPalStat(level); 
        std::cout<<"Naive Color TIme: "<<palStat.invColorTime<<"\n";
    }
    
    std::cout<<"# of Final colors: " <<palcol.getNumColors()<<std::endl;
    if(isValid) {
      if(palcol.checkValidity(jsongraph)) 
        std::cout<<"Coloring valid"<<std::endl;
      else
        std::cout<<"Coloring invalid"<<std::endl;
    }
        
    if(outFname.empty() == false) {
      //Be advised that using this fuction is going to increase memory consumption, since we are recreating the data array. This is only needed if we are encoding the data.
      jsongraph.computeDataArray(true);
      std::vector<NODE_T> cols = palcol.getColors();
      json jsonGrp = jsongraph.createColGroup(cols, palcol.getNumColors()); 
      std::string jsonString = jsonGrp.dump(4);
      std::ofstream outFile(outFname);
      if(outFile.is_open()) {
        outFile << jsonString;
        outFile.close();
      } else {
          std::cerr<< outFname<<" can not be opened" <<std::endl; 
      }
    }
  }
  else{
    std::cout << "Using 64-bit offsets" << std::endl;
    PaletteColor<unsigned long long> palcol(n,target,alpha,list_size,seed);
  
    int level = 0;
    palcol.buildConfGraphGpuMemConscious(jsongraph);
    if(orderName == "RANDOM"){
      palcol.confColorRandCSR();
    }
    else{
      palcol.confColorGreedyCSR();  
    }
    std::vector <NODE_T>  invVert = palcol.getInvVertices();
    PalColStat palStat = palcol.getPalStat(level); 
    palStat.m = jsongraph.getNumEdge();
    palStat.nColors = palcol.getNumColors(); 
    printStat(level,palStat);
    if (isRec == true) {
        while(invVert.size() > nInv) { 
            jsongraph.resetNumEdge();
            level++;
            if(invVert.empty() == false) {
                // if(invVert.size() > 40000) alpha = 3;
                // else if (invVert.size() > 20000) alpha = 2; 
                // else if (invVert.size() > 5000) alpha = 1.5; 
                // else alpha = 1;
                // std::cout <<nextFrac<<std::endl;
                palcol.reInit(invVert,invVert.size()*nextFrac,alpha);
                palcol.buildConfGraphGpuMemConscious(jsongraph,invVert);
                if(orderName == "RANDOM"){
                  palcol.confColorRandCSR(invVert);
                }
                else{
                  palcol.confColorGreedyCSR(invVert);
                }
            }
            invVert = palcol.getInvVertices();
            palStat = palcol.getPalStat(level); 
            palStat.m = jsongraph.getNumEdge();
            //palStat.mConf = palcol.getNumConflicts();
            palStat.nColors = palcol.getNumColors(); 
            printStat(level,palStat);
        }
    }
    if(invVert.empty() == false) {
        std::cout<<"Final Num invalid Vert: "<<invVert.size()<<"\n";
        palcol.naiveGreedyColor<std::vector<uint32_t>>(invVert, jsongraph, palcol.getNumColors());
        palStat = palcol.getPalStat(level); 
        std::cout<<"Naive Color TIme: "<<palStat.invColorTime<<"\n";
    }
    
    std::cout<<"# of Final colors: " <<palcol.getNumColors()<<std::endl;
    if(isValid) {
      if(palcol.checkValidity(jsongraph)) 
        std::cout<<"Coloring valid"<<std::endl;
      else
        std::cout<<"Coloring invalid"<<std::endl;
    }
    
    if(outFname.empty() == false) {
      //Be advised that using this fuction is going to increase memory consumption, since we are recreating the data array. This is only needed if we are encoding the data.
      jsongraph.computeDataArray(true);
      std::vector<NODE_T> cols = palcol.getColors();
      json jsonGrp = jsongraph.createColGroup(cols, palcol.getNumColors()); 
      std::string jsonString = jsonGrp.dump(4);
      std::ofstream outFile(outFname);
      if(outFile.is_open()) {
        outFile << jsonString;
        outFile.close();
      } else {
          std::cerr<< outFname<<" can not be opened" <<std::endl; 
      }
    }
  }


  return 0;
}  

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


#include "ClqPart/JsonGraph.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <string>


namespace ClqPart {
  std::pair<std::complex<double>,std::string> pqMerge(std::string P, std::string Q)
  {
    auto L = P.size();
    EDGE_T nimag = 0; // numberf of 'i'
    EDGE_T nsign = 0; // number of '-1'
    std::string new_string = "";
    for (auto i = 0; i < L; i++)
    {
        if (P[i] == 'I')
        {
            new_string += Q[i];
        }
        else if (Q[i] == 'I')
        {
            new_string += P[i];
        }
        else if (P[i] == Q[i])
        {
            new_string += 'I';
        }
        else if (P[i] == 'X')
        {
            if (Q[i] == 'Y')
            {
                new_string += 'Z';
                nimag += 1;
            }
            else if (Q[i] == 'Z')
            {
                new_string += 'Y';
                nimag += 1;
                nsign += 1;
            }
        }
        else if (P[i] == 'Y')
        {
            if (Q[i] == 'X')
            {
                new_string += 'Z';
                nimag += 1;
                nsign += 1;
            }
            else if (Q[i] == 'Z')
            {
                new_string += 'X';
                nimag += 1;
            }
        }
        else if (P[i] == 'Z')
        {
            if (Q[i] == 'X')
            {
                new_string += 'Y';
                nimag += 1;
            }
            else if (Q[i] == 'Y')
            {
                new_string += 'X';
                nimag += 1;
                nsign += 1;
            }
        }
    }
    //
    std::complex<double> sgn = std::pow(std::complex<double>(0, 1), nimag) * std::pow(-1, nsign);
    //
    return std::make_pair(sgn,new_string);
  }

  void JsonGraph::printData() {
    std::cout<<"data in array"<<std::endl;
    std::cout<<dataAr.dump()<<std::endl; 

    std::cout<<"data in object"<<std::endl;
    std::cout<<data.dump()<<std::endl; 
  }

  bool JsonGraph::is_an_edge(std::string P, std::string Q) {
    int L = P.length();
    int cnt = 0;
    for (int i = 0; i < L; i++) {
      if (P[i] != 'I' && Q[i] != 'I' && P[i] != Q[i]) {
        cnt++;
      }
    }
    if (cnt % 2 == 1) {
      return true;
    }
    else {
      numEdgeCom++;
      return false;
    }
  }

  bool JsonGraph::is_an_edge(const std::vector<uint32_t> &P_List, const std::vector<uint32_t> &Q_List) {
    // transform_reduce P_list and Q_list by bitwise-and and popcount
    uint32_t cnt = 0;
    #pragma omp simd reduction(+:cnt)
    for (int i = 0; i < P_List.size(); i++) {
      cnt += __builtin_popcount(P_List[i] & Q_List[i]);
    }
    if (cnt & 0x1) {
      return true;
    }
    else {
      numEdgeCom++;
      return false;
    }
  }
  #ifdef ENABLE_GPU
  bool JsonGraph::is_an_edge(NODE_T P, NODE_T Q) {
    // transform_reduce P_list and Q_list by bitwise-and and popcount
    uint32_t cnt = 0;
    const int P_offset = P * pauliEncSize;
    const int Q_offset = Q * pauliEncSize;
    for(int i = 0; i < pauliEncSize; i++) {
      cnt += __builtin_popcount(dataEnc[P_offset + i] & dataEnc[Q_offset + i]);
    }
    if (cnt & 0x1) {
      return true;
    }
    else {
      numEdgeCom++;
      return false;
    }
  }
  #endif

  // bool JsonGraph::is_an_edge_alternative(std::vector<uint32_t> P, std::vector<uint32_t> Q, int num_terms) {
  //   int cnt = 0;
  //   for (int i = 0; i < P.size(); i++) {
  //     cnt += __builtin_popcount(P[i] & Q[i]);
  //   }
  //   if (cnt & 0x1) {
  //     return true;
  //   }
  //   else {
  //     numEdgeCom++;
  //     return false;
  //   }
  // } 
  
  bool static fileTypeCheck(std::string fn, std::string extension)
  {
    if(fn.substr(fn.find_last_of(".") + 1) == extension) {
      return true;
    } else {
      return false;
    }
  }

  EDGE_T JsonGraph::calculateNumComEdges() {
  

    NODE_T n=numDataPoints;
    EDGE_T m = 0;

    for (auto eu =0 ;eu<n-1;eu++) {
      for (auto ev=eu+1;ev<n;ev++) {
        if (is_an_edge<std::vector<uint32_t> >(eu,ev)) {
            continue;
        }
        else {
          //std::cout<<Z1.second<<" "<<Z2.second<<" "<<i+1<<" "<<j+1<<"\n";
          m++;
        }
      }
    }
    return m;
  }

  void JsonGraph::ReadJsonAdjacencyGraph() {

    //json::iterator it = data.begin(); 
    
    /*
    for (auto& [key, val] : data.items())
    {
      std::string s(val);
      std::cout << std::fixed<< "key: " << key << ", value:" << std::stod(s) << '\n';
    } 
    */
    
    double t1 = omp_get_wtime();     

    NODE_T n=numDataPoints;
    EDGE_T m = 0;

    for (auto eu =0 ;eu<n-1;eu++) {
      for (auto ev=eu+1;ev<n;ev++) {
        if (is_an_edge<std::vector<uint32_t> >(eu,ev)) {
            continue;
        }
        else {
          //std::cout<<Z1.second<<" "<<Z2.second<<" "<<i+1<<" "<<j+1<<"\n";
          m++;
          Edge e{eu,ev};
          graph.edgeList.push_back(e);
        }
      }
    }

    graph.n = n;
    graph.m = m;
    numEdgeCom = m;
    numDataPoints = n;
    generateTime = omp_get_wtime() - t1;
  }

  /*
  void JsonGraph::advance() {
     
    ++it1;
    if(it1 == data.end())
    {
      ++it; 
      it1 = std::next(it,1);
      u++;
      v=u+1;
    } 
    else {
      v++; 
    }
  } 

  bool JsonGraph::nextEdge( Edge &e ) {
    while(1) {
      if(it == std::prev(data.end(),1)) {
        return false;      
      }
      if (is_an_edge(it.key(),it1.key())) {
        advance();
        continue;
      }
      else {
        e.u = u;
        e.v = v;
        advance();
        numEdge++;
        return true;
      }

    }
     
  }
  */
  void JsonGraph::nextIndices() {
    v++;
    if(v>=numDataPoints) {
      u++;
      v = u + 1; 
    } 
  }

  void JsonGraph::writeGraphMtx(std::string fileName) {
    if(fileTypeCheck(fileName,"mtx")==false)
    {
        std::cout << "file type is not mtx"<<std::endl;
        std::exit(1);
    }
    std::ofstream myfile(fileName.c_str());

    double t1 = omp_get_wtime();     
    if(myfile.is_open())
    {
        myfile << "%%MatrixMarket matrix coordinate pattern symmetric"<<std::endl;
        myfile << graph.n<<" "<<graph.n<<" "<<graph.m<<"\n";
        for(auto elem: graph.edgeList) { 
          myfile<< elem.v+1<<" "<<elem.u+1<<"\n";
        }
        myfile.close();
    }
    else
        std::cout<<"unable to open file"<<std::endl;

    writeTime = omp_get_wtime() - t1;
     
  }
  void JsonGraph::ReadConstructWriteGraph(std::string fileName) {

    if(fileTypeCheck(fileName,"mtx")==false) {
      std::cout << "file type is not mtx"<<std::endl;
      std::exit(1);
    }

    std::ofstream myfile(fileName.c_str());

    if(myfile.is_open() == false) {
      std::cout<<"unable to open file"<<std::endl; 
      exit(1);
    }
    
    double t1 = omp_get_wtime();     
    NODE_T n=numDataPoints;
    EDGE_T m = 0;

    for (auto eu =0 ;eu<n-1;eu++)
    {
      for (auto ev=eu+1;ev<n;ev++) {
        if (is_an_edge<std::vector<uint32_t> >(eu,ev)) {
            continue;
        }
        else {
          //std::cout<<Z1.second<<" "<<Z2.second<<" "<<i+1<<" "<<j+1<<"\n";
          m++;
          myfile<<ev+1<<" "<<eu+1<<"\n";
        }
      }
    }
    myfile<<n<<" "<<n<<" "<<m<<std::endl;
    generateTime = omp_get_wtime() - t1;
  }


  json JsonGraph::createColGroup(std::vector<NODE_T> &colors, NODE_T nColors) {

    NODE_T n1 = colors.size();
    /*
    std::vector< std::vector<std::pair<std::string,std::string> > > dict(nColors);

    for(auto u = 0; u<n1; u++) {
      dict[colors[u]].push_back(std::make_pair(dataAr[u][0],dataAr[u][1])); 
    }

    */
    json jsonGrp;

    /*
    for(auto c=0;c<nColors;c++) {
      jsonGrp[c] = dict[c]; 
    }
    */
    for(auto u = 0; u<n1; u++) {
      jsonGrp[std::to_string(colors[u])]["paulis"].push_back(dataAr[u][0]);
      jsonGrp[std::to_string(colors[u])]["coeffs"].push_back(dataAr[u][1]);
    }
    return jsonGrp;
  }
}

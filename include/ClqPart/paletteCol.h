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

#pragma once

// #define ENABLE_GPU
// #define COMPLEMENT_GRAPH

#include "ClqPart/graph.h" 
#include "ClqPart/JsonGraph.h"
#include <random>

#include <omp.h>

#ifdef ENABLE_GPU
// #include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdlib.h>
#include <atomic>
// CUDA Error Checking Code
#define ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
        file, line);
    if (abort)
      exit(code);
  }
}
#include "ClqPart/cuPaletteCol.cuh"
#endif

bool findFirstCommonElement(const std::vector<NODE_T>& vec1, const std::vector<NODE_T>& vec2);

struct PalColStat {
  NODE_T n;
  EDGE_T m;
  EDGE_T mConf;
  NODE_T palSz;
  NODE_T lstSz;
  NODE_T nColors;
  double assignTime;
  double confBuildTime;
  double confColorTime; 
  double invColorTime;
};

template<typename OffsetTy = int>
class PaletteColor {
  
  NODE_T n;
  NODE_T colThreshold;
  NODE_T T;
  int seed;
  std::vector<std::vector<NODE_T> > colList;
  std::vector<NODE_T> colors;
  NODE_T nColors;
  OffsetTy nConflicts;
  int level;

  #ifdef ENABLE_GPU
  std::vector<NODE_T> h_colList;
  NODE_T *d_colList;
  NODE_T *d_colors;
  NODE_T *d_confColors;
  std::vector<uint32_t> h_pauliEnc;
  uint32_t *d_pauliEnc;
  size_t pauliEncSize;
  std::vector<NODE_T> h_confVertices;
  NODE_T *d_confVertices;
  std::vector<OffsetTy> h_confOffsets;
  OffsetTy *d_confOffsets;
  #endif
  
  std::vector<NODE_T> invalidVertices;
  std::vector<NODE_T> vertexOrder;
  std::vector<std::vector< NODE_T> > confAdjList;
  std::vector<PalColStat> palStat;

public:
  PaletteColor( NODE_T n1, NODE_T colThresh, float alpha=1, NODE_T lst_sz = -1, int my_seed=123) {
    n = n1;
    colThreshold = colThresh; 
    colors.resize(n,-1);
    nColors = 0;
    nConflicts = 0;
    if(lst_sz < 0)
      T =  static_cast<NODE_T> (alpha*log(n));
    else
      T = lst_sz;

    if(T>colThresh)
      T = colThresh;

    confAdjList.resize(n);
    colList.resize(n);
    invalidVertices.clear();
    palStat.push_back({n,-1,-1,colThreshold,T,0,0.0,0.0,0.0,0.0});
    level = 0;
    seed = my_seed;
    assignListColor();
  }

  //overloaded method
  PaletteColor(NODE_T n1) {
    n=n1; 
    colors.resize(n,-1);
    palStat.push_back({n,-1,-1,-1,-1,0,0.0,0.0,0.0,0.0});
    level=0;
  }

  template<typename PauliTy = std::string>
  void naiveGreedyColor(std::vector<NODE_T> vertList, ClqPart::JsonGraph &jsongraph, NODE_T offset) {

    if(vertList.empty() == false) {
      double t1 = omp_get_wtime();

      std::vector<NODE_T> forbiddenCol(n,-1);
      colors[vertList[0]] = offset; 

      for(auto i=1; i<vertList.size();i++) {
        NODE_T eu = vertList[i]; 
        for(auto j=0; j<i; j++) {
          NODE_T ev = vertList[j]; 

          //std::cout<<eu<<std::endl;
          if(jsongraph.is_an_edge<PauliTy>(eu,ev) == false) { 
            if (colors[ev] >= 0) {
              forbiddenCol[colors[ev]] = eu;
            }
          } 
        }
      
        //color eu with first available color
        for( auto ii=offset;ii<n;ii++) {
          if(forbiddenCol[ii] == -1) {
            colors[eu] = ii; 
            break;
          } 
        }
      }
      palStat[level].invColorTime = omp_get_wtime() - t1; 
      nColors = *std::max_element(colors.begin(),colors.end()) + 1;
    }
  }

  //This function computes the graph directly, rather in streaming way. It takes
//a JsonGraph object since it requires to determine whether the pair (eu,ev) is 
//an edge in the complement graph.
template<typename PauliTy = std::string>
void buildConfGraph ( ClqPart::JsonGraph &jsongraph) {

  double t1 = omp_get_wtime();
  for(NODE_T eu =0; eu < n-1; eu++) {
    for(NODE_T ev = eu+1; ev < n; ev++) {

      if(jsongraph.is_an_edge<PauliTy>(eu,ev) == false) {
        bool hasCommon = findFirstCommonElement(colList[eu],colList[ev]);
        if(hasCommon == true ) {

          confAdjList[eu].push_back(ev); 
          confAdjList[ev].push_back(eu); 
          nConflicts++;
        }
      }
    }
  }
  palStat[level].confBuildTime = omp_get_wtime() - t1;
  palStat[level].mConf = nConflicts;
}

//Overloaded function to work on the subset of the nodes
template<typename PauliTy = std::string>
void buildConfGraph ( ClqPart::JsonGraph &jsongraph, std::vector<NODE_T> &nodeList) {

  double t1 = omp_get_wtime();
  NODE_T num = nodeList.size(); 
  for(NODE_T iu =0; iu < num-1; iu++) {
    for(NODE_T iv = iu+1; iv < num; iv++) {
      auto eu = nodeList[iu];
      auto ev = nodeList[iv];

      if(jsongraph.is_an_edge<PauliTy>(eu,ev) == false) {
        // Assert colList[eu] and colList[ev] sizes are equal to T
        bool hasCommon = findFirstCommonElement(colList[eu],colList[ev]);
        if(hasCommon == true ) {

          confAdjList[eu].push_back(ev); 
          confAdjList[ev].push_back(eu); 
          nConflicts++;
        }
      }
    }
  }
  palStat[level].confBuildTime = omp_get_wtime() - t1;
  palStat[level].mConf = nConflicts;

}


//This function checks whether the edge (eu,ev) (in complement graph) is a conflict. NOte that 
//this is for constructing the graph in streaming way, so we are presented with
//one edge at a time. (eu,ev) is an edge in a complement graph.
void buildStreamConfGraph ( NODE_T eu, NODE_T ev) {

  bool hasCommon = findFirstCommonElement(colList[eu],colList[ev]);
  if(hasCommon == true ) {

    confAdjList[eu].push_back(ev); 
    confAdjList[ev].push_back(eu); 
    nConflicts++;
    palStat[level].mConf++;
  }


}

#ifdef ENABLE_GPU
void buildConfGraphGpuMemConscious (ClqPart::JsonGraph &jsongraph) {

  double t1 = omp_get_wtime();

  pauliEncSize = jsongraph.getPauliEncSize();
  std::vector<uint32_t> &h_pauliEnc = jsongraph.getEncodedData();
  ERR_CHK(cudaMalloc(&d_pauliEnc, h_pauliEnc.size() * sizeof(NODE_T)));
  ERR_CHK(cudaMemcpy(d_pauliEnc, h_pauliEnc.data(), h_pauliEnc.size() * sizeof(NODE_T), cudaMemcpyHostToDevice));

  h_confOffsets.resize(n+1);
  ERR_CHK(cudaMalloc(&d_confOffsets, h_confOffsets.size() * sizeof(OffsetTy)));
  ERR_CHK(cudaMemset(d_confOffsets, 0, h_confOffsets.size() * sizeof(OffsetTy)));

  OffsetTy *d_nConflicts;
  ERR_CHK(cudaMalloc(&d_nConflicts, sizeof(OffsetTy)));
  ERR_CHK(cudaMemset(d_nConflicts, 0, sizeof(OffsetTy)));

  #ifndef COMPLEMENT_GRAPH

  // Find out how much free memory is on the GPU
  size_t freeMem, totalMem;
  ERR_CHK(cudaMemGetInfo(&freeMem, &totalMem));
  size_t worst_size = n * (n - 1) * sizeof(NODE_T) * 2;
  // Allocate 90% of it
  size_t allocMem = std::min((size_t)(freeMem * 0.95), worst_size);
  ERR_CHK(cudaMalloc(&d_confVertices, allocMem));

  ERR_CHK(cudaDeviceSynchronize());

  buildCooConfGraphDevice(d_pauliEnc, pauliEncSize, d_colList, n, T, d_confOffsets, d_confVertices, d_nConflicts);
  ERR_CHK(cudaDeviceSynchronize());
  // Read d_nConflicts from GPU
  ERR_CHK(cudaMemcpy(&nConflicts, d_nConflicts, sizeof(OffsetTy), cudaMemcpyDeviceToHost));
  h_confVertices.resize(nConflicts*2);
  OffsetTy *d_confOffsetsCnt = cubExclusiveSum(n, d_confOffsets);
  cudaDeviceSynchronize();
  // if nConflicts * sizeof(NODE_T) < half of allocMem, then we can use the memory
  // allocated for d_confVertices
  if(nConflicts * 2 < (allocMem/sizeof(NODE_T))/2){
    std::cout << "Fits: " << nConflicts * 2 * sizeof(NODE_T) << " < " << allocMem/2 << std::endl;
    NODE_T *d_confCsr = d_confVertices + nConflicts*2;
    buildCsrConfGraphDevice(n, d_confOffsets, d_confOffsetsCnt, d_confVertices, d_confCsr, nConflicts);
    ERR_CHK(cudaDeviceSynchronize());
    // double t2 = omp_get_wtime();
    // Read d_confCsr from GPU
    ERR_CHK(cudaMemcpy(h_confOffsets.data(), d_confOffsets, h_confOffsets.size() * sizeof(OffsetTy), cudaMemcpyDeviceToHost));
    ERR_CHK(cudaMemcpy(h_confVertices.data(), d_confCsr, h_confVertices.size() * sizeof(NODE_T), cudaMemcpyDeviceToHost));
    ERR_CHK(cudaDeviceSynchronize());
    // std:: cout << "Copy time: " << omp_get_wtime() - t2 << std::endl;
    // std::cout << h_confVertices[0] << " " << h_confVertices[nConflicts*2-1] << std::endl;
  }
  else{
    std::cout << "Doesn't fit: " << nConflicts * 2 * sizeof(NODE_T) << " > " << allocMem/2 << std::endl;
    // Too Large for GPU memory, use CPU to post-process
    ERR_CHK(cudaMemcpy(h_confOffsets.data(), d_confOffsets, h_confOffsets.size() * sizeof(OffsetTy), cudaMemcpyDeviceToHost));
    std::vector<Edge> h_cooVerticesTmp(nConflicts);
    ERR_CHK(cudaMemcpy(h_cooVerticesTmp.data(), d_confVertices, h_cooVerticesTmp.size() * sizeof(Edge), cudaMemcpyDeviceToHost));
    std::vector<std::atomic<OffsetTy>> h_confOffsetsTmp(n);
    for(NODE_T i = 0; i < n; i++){
      std::atomic_init(&h_confOffsetsTmp[i], 0);
    }
    #pragma omp parallel for
    for(OffsetTy i = 0; i < nConflicts; i++){
      Edge e = h_cooVerticesTmp[i];
      OffsetTy offset = h_confOffsets[e.u] + std::atomic_fetch_add(&h_confOffsetsTmp[e.u], (OffsetTy)1);
      h_confVertices[offset] = e.v;
      offset = h_confOffsets[e.v] + std::atomic_fetch_add(&h_confOffsetsTmp[e.v], (OffsetTy)1);
      h_confVertices[offset] = e.u;
    }
  }
  ERR_CHK(cudaFree(d_confOffsetsCnt));
  ERR_CHK(cudaDeviceSynchronize());
  // std::cout << "nConflicts: " << nConflicts << std::endl;
  #pragma omp parallel for
  for(NODE_T i = 0; i < n; i++) {
    // Sort each vertex's edgelist
    std::sort(h_confVertices.begin() + h_confOffsets[i], h_confVertices.begin() + h_confOffsets[i+1]);
  }
  palStat[level].confBuildTime = omp_get_wtime() - t1;
  palStat[level].mConf = nConflicts;
  #else // COMPLEMENT_GRAPH
  // Compute what the offsets would be for a complement graph
  buildCooCompGraphDevice(d_pauliEnc, pauliEncSize, d_colList, n, T, d_confOffsets, d_nConflicts);
  ERR_CHK(cudaDeviceSynchronize());
  ERR_CHK(cudaMemcpy(&nConflicts, d_nConflicts, sizeof(OffsetTy), cudaMemcpyDeviceToHost));
  ERR_CHK(cudaMemcpy(h_confOffsets.data(), d_confOffsets, h_confOffsets.size() * sizeof(OffsetTy), cudaMemcpyDeviceToHost));
  ERR_CHK(cudaDeviceSynchronize());

  // Find lowest degree and highest degree vertex
  NODE_T minDegree = n;
  NODE_T maxDegree = 0;
  double avgDegree = 0;
  for(NODE_T i = 0; i < n; i++) {
    OffsetTy degree = h_confOffsets[i];
    if(degree < minDegree) {
      minDegree = degree;
    }
    if(degree > maxDegree) {
      maxDegree = degree;
    }
    avgDegree += degree;
  }
  avgDegree /= n;
  // Calculate variance
  double variance = 0;
  for(NODE_T i = 0; i < n; i++) {
    OffsetTy degree = h_confOffsets[i];
    variance += (degree - avgDegree) * (degree - avgDegree);
  }
  variance /= n;
  // Print values
  std::cout << "vertices: " << n << std::endl;
  std::cout << "edges: " << nConflicts << std::endl;
  std::cout << "pauli mtx per string: " << jsongraph.getPauliLength() << std::endl;
  std::cout << "minDegree: " << minDegree << std::endl;
  std::cout << "maxDegree: " << maxDegree << std::endl;
  std::cout << "avgDegree: " << avgDegree << std::endl;
  std::cout << "variance: " << variance << std::endl;
  // Exit
  exit(0);
  #endif // COMPLEMENT_GRAPH
}

// Build conflict graph on GPU from an input CSR (ECL) graph.
// rowPtr has size n+1; colIdx has size numAdj (total adjacency entries).
void buildConfGraphGpuFromCSR(const int *rowPtr, const int *colIdx, int nVertices, int numAdj) {
  double t1 = omp_get_wtime();
  // Upload input CSR to device
  int *d_rowPtr = nullptr; int *d_colIdx = nullptr;
  ERR_CHK(cudaMalloc(&d_rowPtr, (nVertices + 1) * sizeof(int)));
  ERR_CHK(cudaMemcpy(d_rowPtr, rowPtr, (nVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
  ERR_CHK(cudaMalloc(&d_colIdx, numAdj * sizeof(int)));
  ERR_CHK(cudaMemcpy(d_colIdx, colIdx, numAdj * sizeof(int), cudaMemcpyHostToDevice));

  // Allocate offsets and COO buffer on device
  h_confOffsets.resize(n + 1);
  ERR_CHK(cudaMalloc(&d_confOffsets, h_confOffsets.size() * sizeof(OffsetTy)));
  ERR_CHK(cudaMemset(d_confOffsets, 0, h_confOffsets.size() * sizeof(OffsetTy)));
  // Allocate max possible COO edges buffer (worst-case: every input edge conflicts)
  ERR_CHK(cudaMalloc(&d_confVertices, (size_t)numAdj * sizeof(Edge)));

  // Count edges
  OffsetTy *d_nConflicts = nullptr;
  ERR_CHK(cudaMalloc(&d_nConflicts, sizeof(OffsetTy)));
  ERR_CHK(cudaMemset(d_nConflicts, 0, sizeof(OffsetTy)));

  buildCooConfGraphFromCSRDevice(d_rowPtr, d_colIdx, (NODE_T)n, d_colList, (NODE_T)T, d_confOffsets, d_confVertices, d_nConflicts);
  ERR_CHK(cudaDeviceSynchronize());
  ERR_CHK(cudaMemcpy(&nConflicts, d_nConflicts, sizeof(OffsetTy), cudaMemcpyDeviceToHost));

  // Prepare CSR of conflict graph
  h_confVertices.resize((size_t)nConflicts * 2);
  OffsetTy *d_confOffsetsCnt = cubExclusiveSum((NODE_T)n, d_confOffsets);
  cudaDeviceSynchronize();
  NODE_T *d_confCsr = nullptr;
  ERR_CHK(cudaMalloc(&d_confCsr, h_confVertices.size() * sizeof(NODE_T)));
  buildCsrConfGraphDevice((NODE_T)n, d_confOffsets, d_confOffsetsCnt, d_confVertices, d_confCsr, nConflicts);
  ERR_CHK(cudaDeviceSynchronize());

  // Copy CSR back
  ERR_CHK(cudaMemcpy(h_confOffsets.data(), d_confOffsets, h_confOffsets.size() * sizeof(OffsetTy), cudaMemcpyDeviceToHost));
  ERR_CHK(cudaMemcpy(h_confVertices.data(), d_confCsr, h_confVertices.size() * sizeof(NODE_T), cudaMemcpyDeviceToHost));

  // Cleanup temps
  ERR_CHK(cudaFree(d_confOffsetsCnt));
  ERR_CHK(cudaFree(d_confCsr));
  ERR_CHK(cudaFree(d_rowPtr));
  ERR_CHK(cudaFree(d_colIdx));
  ERR_CHK(cudaFree(d_nConflicts));

  // Sort adjacency lists on host
  #pragma omp parallel for
  for (NODE_T i = 0; i < n; i++) {
    std::sort(h_confVertices.begin() + h_confOffsets[i], h_confVertices.begin() + h_confOffsets[i + 1]);
  }
  palStat[level].confBuildTime = omp_get_wtime() - t1;
  palStat[level].mConf = nConflicts;
}

void buildConfGraphGpuMemConscious (ClqPart::JsonGraph &jsongraph, std::vector<NODE_T> &nodeList) {

  double t1 = omp_get_wtime();

  // Write nodeList to GPU
  NODE_T *d_nodeList;
  ERR_CHK(cudaMalloc(&d_nodeList, nodeList.size() * sizeof(NODE_T)));
  ERR_CHK(cudaMemcpy(d_nodeList, nodeList.data(), nodeList.size() * sizeof(NODE_T), cudaMemcpyHostToDevice));

  pauliEncSize = jsongraph.getPauliEncSize();
  // std::vector<uint32_t> &h_pauliEnc = jsongraph.getEncodedData();
  // ERR_CHK(cudaMalloc(&d_pauliEnc, h_pauliEnc.size() * sizeof(NODE_T)));
  // ERR_CHK(cudaMemcpy(d_pauliEnc, h_pauliEnc.data(), h_pauliEnc.size() * sizeof(NODE_T), cudaMemcpyHostToDevice));

  // h_confOffsets.resize(n+1);
  // ERR_CHK(cudaMalloc(&d_confOffsets, h_confOffsets.size() * sizeof(OffsetTy)));
  ERR_CHK(cudaMemset(d_confOffsets, 0, h_confOffsets.size() * sizeof(OffsetTy)));

  OffsetTy *d_nConflicts;
  ERR_CHK(cudaMalloc(&d_nConflicts, sizeof(OffsetTy)));
  ERR_CHK(cudaMemset(d_nConflicts, 0, sizeof(OffsetTy)));
  // Find out how much free memory is on the GPU
  size_t freeMem, totalMem;
  ERR_CHK(cudaMemGetInfo(&freeMem, &totalMem));
  size_t worst_size = nodeList.size() * (nodeList.size() - 1) * sizeof(NODE_T) * 2;
  // Allocate 90% of it
  size_t allocMem = std::min((size_t)(freeMem * 0.95), worst_size);
  ERR_CHK(cudaMalloc(&d_confVertices, allocMem));

  ERR_CHK(cudaDeviceSynchronize());

  buildCooConfGraphDevice(d_pauliEnc, pauliEncSize, d_colList, d_nodeList, (NODE_T)nodeList.size(), T, d_confOffsets, d_confVertices, d_nConflicts);
  ERR_CHK(cudaDeviceSynchronize());
  // Read d_nConflicts from GPU
  ERR_CHK(cudaMemcpy(&nConflicts, d_nConflicts, sizeof(OffsetTy), cudaMemcpyDeviceToHost));
  h_confVertices.resize(nConflicts*2);
  OffsetTy *d_confOffsetsCnt = cubExclusiveSum(n, d_confOffsets);
  cudaDeviceSynchronize();
  // if nConflicts * sizeof(NODE_T) < half of allocMem, then we can use the memory
  // allocated for d_confVertices
  if(nConflicts * 2 < (allocMem/sizeof(NODE_T))/2){
    std::cout << "Fits: " << nConflicts * 2 * sizeof(NODE_T) << " < " << allocMem/2 << std::endl;
    NODE_T *d_confCsr = d_confVertices + nConflicts*2;
    buildCsrConfGraphDevice(n, d_confOffsets, d_confOffsetsCnt, d_confVertices, d_confCsr, nConflicts);
    ERR_CHK(cudaDeviceSynchronize());
    // double t2 = omp_get_wtime();
    // Read d_confCsr from GPU
    ERR_CHK(cudaMemcpy(h_confOffsets.data(), d_confOffsets, h_confOffsets.size() * sizeof(OffsetTy), cudaMemcpyDeviceToHost));
    ERR_CHK(cudaMemcpy(h_confVertices.data(), d_confCsr, h_confVertices.size() * sizeof(NODE_T), cudaMemcpyDeviceToHost));
    ERR_CHK(cudaDeviceSynchronize());
    // std:: cout << "Copy time: " << omp_get_wtime() - t2 << std::endl;
    // std::cout << h_confVertices[0] << " " << h_confVertices[nConflicts*2-1] << std::endl;
  }
  else{
    // Too Large for GPU memory, use CPU to post-process
    ERR_CHK(cudaMemcpy(h_confOffsets.data(), d_confOffsets, h_confOffsets.size() * sizeof(OffsetTy), cudaMemcpyDeviceToHost));
    std::vector<Edge> h_cooVerticesTmp(nConflicts);
    ERR_CHK(cudaMemcpy(h_cooVerticesTmp.data(), d_confVertices, h_cooVerticesTmp.size() * sizeof(Edge), cudaMemcpyDeviceToHost));
    std::vector<std::atomic<OffsetTy>> h_confOffsetsTmp(n);
    for(NODE_T i = 0; i < nodeList.size(); i++){
      std::atomic_init(&h_confOffsetsTmp[i], 0);
    }
    #pragma omp parallel for
    for(NODE_T i = 0; i < nConflicts; i++){
      Edge e = h_cooVerticesTmp[i];
      OffsetTy offset = h_confOffsets[e.u] + std::atomic_fetch_add(&h_confOffsetsTmp[e.u], 1);
      h_confVertices[offset] = e.v;
      offset = h_confOffsets[e.v] + std::atomic_fetch_add(&h_confOffsetsTmp[e.v], 1);
      h_confVertices[offset] = e.u;
    }
  }
  // Free d_confOffsetsCnt
  ERR_CHK(cudaFree(d_confOffsetsCnt));
  ERR_CHK(cudaFree(d_nodeList));
  ERR_CHK(cudaFree(d_nConflicts));
  // std::cout << "nConflicts: " << nConflicts << std::endl;
  #pragma omp parallel for
  for(NODE_T i : nodeList) {
    // std::cout << "Sorting " << i << std::endl;
    // Sort each vertex's edgelist
    std::sort(h_confVertices.begin() + h_confOffsets[i], h_confVertices.begin() + h_confOffsets[i+1]);
  }
  palStat[level].confBuildTime = omp_get_wtime() - t1;
  palStat[level].mConf = nConflicts;
}

void buildConfGraphGpu (ClqPart::JsonGraph &jsongraph) {
  double t1 = omp_get_wtime();
  pauliEncSize = jsongraph.getPauliEncSize();
  std::vector<uint32_t> &h_pauliEnc = jsongraph.getEncodedData();
  ERR_CHK(cudaMalloc(&d_pauliEnc, h_pauliEnc.size() * sizeof(NODE_T)));
  ERR_CHK(cudaMemcpy(d_pauliEnc, h_pauliEnc.data(), h_pauliEnc.size() * sizeof(NODE_T), cudaMemcpyHostToDevice));

  h_confOffsets.resize(n);
  ERR_CHK(cudaMalloc(&d_confOffsets, h_confOffsets.size() * sizeof(OffsetTy)));
  #ifndef COMPLEMENT_GRAPH
  h_confVertices.resize(n * n);
  ERR_CHK(cudaMalloc(&d_confVertices, h_confVertices.size() * sizeof(NODE_T)));
  #endif // !COMPLEMENT_GRAPH

  ERR_CHK(cudaDeviceSynchronize());

  // Create d_nConflicts and initialize to 0
  OffsetTy *d_nConflicts;
  ERR_CHK(cudaMalloc(&d_nConflicts, sizeof(OffsetTy)));
  ERR_CHK(cudaMemset(d_nConflicts, 0, sizeof(OffsetTy)));

  // Call function to send to GPU to build conf graph
  #ifdef COMPLEMENT_GRAPH
  buildCompGraphDevice(d_pauliEnc, pauliEncSize, d_colList, n, T, d_confOffsets, d_confVertices, d_nConflicts);
  #else // !COMPLEMENT_GRAPH
  buildConfGraphDevice(d_pauliEnc, pauliEncSize, d_colList, n, T, d_confOffsets, d_confVertices, d_nConflicts);
  #endif // !COMPLEMENT_GRAPH
  ERR_CHK(cudaDeviceSynchronize());
  // Read d_nConflicts from GPU
  ERR_CHK(cudaMemcpy(&nConflicts, d_nConflicts, sizeof(OffsetTy), cudaMemcpyDeviceToHost));
  // Free d_nConflicts
  ERR_CHK(cudaFree(d_nConflicts));
  // Print nConflicts
  nConflicts /= 2;
  // Copy h_confOffsets and h_confVertices from GPU
  ERR_CHK(cudaMemcpy(h_confOffsets.data(), d_confOffsets, h_confOffsets.size() * sizeof(OffsetTy), cudaMemcpyDeviceToHost));
  ERR_CHK(cudaMemcpy(h_confVertices.data(), d_confVertices, h_confVertices.size() * sizeof(NODE_T), cudaMemcpyDeviceToHost));

  ERR_CHK(cudaDeviceSynchronize());
  std::cout << "nConflicts: " << nConflicts << std::endl;
  // Exit program
  for(NODE_T i = 0; i < n; i++) {
    confAdjList[i].reserve(h_confOffsets[i]);
    for(NODE_T j = 0; j < h_confOffsets[i]; j++) {
      confAdjList[i].push_back(h_confVertices[i * n + j]);
    }
    std::sort(confAdjList[i].begin(), confAdjList[i].end());
  }
  palStat[level].confBuildTime = omp_get_wtime() - t1;
  palStat[level].mConf = nConflicts;
  #ifdef COMPLEMENT_GRAPH
  // Find lowest degree and highest degree vertex
  NODE_T minDegree = n;
  NODE_T maxDegree = 0;
  double avgDegree = 0;
  for(NODE_T i = 0; i < n; i++) {
    if(h_confOffsets[i] < minDegree) {
      minDegree = h_confOffsets[i];
    }
    if(h_confOffsets[i] > maxDegree) {
      maxDegree = h_confOffsets[i];
    }
    avgDegree += h_confOffsets[i];
  }
  avgDegree /= n;
  // Calculate variance
  double variance = 0;
  for(NODE_T i = 0; i < n; i++) {
    variance += (h_confOffsets[i] - avgDegree) * (h_confOffsets[i] - avgDegree);
  }
  variance /= n;
  // Print values
  std::cout << "vertices: " << n << std::endl;
  std::cout << "edges: " << nConflicts << std::endl;
  std::cout << "pauli mtx per string: " << jsongraph.getPauliLength() << std::endl;
  std::cout << "minDegree: " << minDegree << std::endl;
  std::cout << "maxDegree: " << maxDegree << std::endl;
  std::cout << "avgDegree: " << avgDegree << std::endl;
  std::cout << "variance: " << variance << std::endl;
  // Exit
  exit(0);
#endif // COMPLEMENT_GRAPH
}

void confColorGreedyCSR() {
  
  double t1 = omp_get_wtime();
  // std::cout<<"# of conflicting edges: "<<nConflicts<<std::endl; 
  //Buckets of size T;
  std::vector<std::vector<NODE_T> > verBucket(T+1);
  //to stoe the position of vertex in the corresponding bucket
  std::vector<NODE_T> verLocation(n);
  
  NODE_T vMin = T; 

  std::mt19937 engine(seed);
  std::uniform_int_distribution<NODE_T> uniform_dist(0, colThreshold-1);

  //# of verties processed. For stopping condition later
  NODE_T vtxProcessed = 0;
  for(NODE_T i=0;i<n;i++) {
    //if this is a non-conflicting vertex we can color it arbitrarily from the 
    //list of colors.
    if(h_confOffsets[i+1] - h_confOffsets[i] == 0) {
      //std::cout<<"coloring non-conflicting edge"<<std::endl;
      uniform_dist = std::uniform_int_distribution<NODE_T>(0, 
          colList[i].size()-1);
      colors[i] = colList[i][uniform_dist(engine)];
      vtxProcessed++;
      continue;
    }
    //otherwise place it in appropriate bucket
    NODE_T t = colList[i].size();
    verBucket[t-1].push_back(i); 
    verLocation[i] = verBucket[t-1].size()-1;
    
    //the minimum bucket length
    if(t < vMin) {
      vMin = t; 
    }
  }
 
  //keep processing until we see all the vertices. 
  // double buckets_time = 0.0;
  while (vtxProcessed < n) {
    for(NODE_T i = vMin-1; i < T;i++) {
      //if this bucket has elements
      //select a random vertex and swap it with the last element
      //to efficiently remove this vertex from the bucket.
      //std::cout<<"Bucket: "<<i<<"size: "<<verBucket[i].size()<<std::endl;
      if(verBucket[i].empty() == false) {
        uniform_dist = std::uniform_int_distribution<NODE_T>(0, 
            verBucket[i].size()-1);
        NODE_T selectedVtxLoc = uniform_dist(engine);
        NODE_T selectedVtx = verBucket[i].at(selectedVtxLoc);

        //update the verLocation array of the last element
        verLocation[verBucket[i].back()] = selectedVtxLoc;
        
        //swap the vertex with the last element
        std::swap(verBucket[i][selectedVtxLoc],verBucket[i].back());
        //remove the vertex
        verBucket[i].pop_back();

        //attempt to color this vertex
        NODE_T colInd = attemptToColor(selectedVtx);
        if(colInd>=0) {
          //remove col at colInd of the vtx is not necessary
          //std::swap(colList[selectedVtx][colInd],colList[selectedVtx].back());
          //colList[selectedVtx].pop_back();
          // double bucket_time = omp_get_wtime();
          fixBucketsCSR(selectedVtx,colors[selectedVtx],vtxProcessed,verBucket,verLocation,vMin); 
          // buckets_time += omp_get_wtime() - bucket_time;
          //std::cout<<"updated VMin: "<<vMin<<std::endl;
        }
        else {
          colors[selectedVtx] = -2;
          invalidVertices.push_back(selectedVtx); 
        }
        vtxProcessed++;
        break;
      } 
    }
  }
  // std::cout << "Buckets time: " << buckets_time << std::endl;
  // std::cout << "Mem access time: " << mem_access_time << std::endl;
  // std::cout << "Sacrificial counter: " << sacrificial_counter << std::endl;
  // std::cout<<"# of invalid vertices: "<<invalidVertices.size()
  //   <<std::endl;

 /* 
  NODE_T invalid = 0;
  for(NODE_T i=0;i<n;i++) {
    if(colors[i] == -2)
     invalid++; 
  }
  std::cout<<invalid<<std::endl;
  */

  palStat[level].confColorTime = omp_get_wtime() - t1;
  
  // std::cout<<"Conflict Coloring Time: "<<confColorTime<<std::endl;

  nColors = *std::max_element(colors.begin(),colors.end()) + 1;

}

void confColorGreedyCSR(std::vector<NODE_T> &nodeList) {
  
  double t1 = omp_get_wtime();
  // std::cout<<"# of conflicting edges: "<<nConflicts<<std::endl; 
  //Buckets of size T;
  std::vector<std::vector<NODE_T> > verBucket(T+1);
  //to stoe the position of vertex in the corresponding bucket
  std::vector<NODE_T> verLocation(n);
  
  NODE_T vMin = T; 

  std::mt19937 engine(seed);
  std::uniform_int_distribution<NODE_T> uniform_dist(0, colThreshold-1);

  //# of verties processed. For stopping condition later
  NODE_T vtxProcessed = 0;
  for(NODE_T i:nodeList) {
    //if this is a non-conflicting vertex we can color it arbitrarily from the 
    //list of colors.
    if(h_confOffsets[i+1] - h_confOffsets[i] == 0) {
      //std::cout<<"coloring non-conflicting edge"<<std::endl;
      uniform_dist = std::uniform_int_distribution<NODE_T>(0, 
          colList[i].size()-1);
      colors[i] = colList[i][uniform_dist(engine)];
      vtxProcessed++;
      continue;
    }
    //otherwise place it in appropriate bucket
    NODE_T t = colList[i].size();
    verBucket[t-1].push_back(i); 
    verLocation[i] = verBucket[t-1].size()-1;
    
    //the minimum bucket length
    if(t < vMin) {
      vMin = t; 
    }
  }
 
  //keep processing until we see all the vertices. 
  // double buckets_time = 0.0;
  while (vtxProcessed < nodeList.size()) {
    for(NODE_T i = vMin-1; i < T;i++) {
      //if this bucket has elements
      //select a random vertex and swap it with the last element
      //to efficiently remove this vertex from the bucket.
      //std::cout<<"Bucket: "<<i<<"size: "<<verBucket[i].size()<<std::endl;
      if(verBucket[i].empty() == false) {
        uniform_dist = std::uniform_int_distribution<NODE_T>(0, 
            verBucket[i].size()-1);
        NODE_T selectedVtxLoc = uniform_dist(engine);
        NODE_T selectedVtx = verBucket[i].at(selectedVtxLoc);

        //update the verLocation array of the last element
        verLocation[verBucket[i].back()] = selectedVtxLoc;
        
        //swap the vertex with the last element
        std::swap(verBucket[i][selectedVtxLoc],verBucket[i].back());
        //remove the vertex
        verBucket[i].pop_back();

        //attempt to color this vertex
        NODE_T colInd = attemptToColor(selectedVtx);
        if(colInd>=0) {
          //remove col at colInd of the vtx is not necessary
          //std::swap(colList[selectedVtx][colInd],colList[selectedVtx].back());
          //colList[selectedVtx].pop_back();
          // double bucket_time = omp_get_wtime();
          fixBucketsCSR(selectedVtx,colors[selectedVtx],vtxProcessed,verBucket,verLocation,vMin); 
          // buckets_time += omp_get_wtime() - bucket_time;
          //std::cout<<"updated VMin: "<<vMin<<std::endl;
        }
        else {
          colors[selectedVtx] = -2;
          invalidVertices.push_back(selectedVtx); 
        }
        vtxProcessed++;
        break;
      } 
    }
  }
  // std::cout << "Buckets time: " << buckets_time << std::endl;
  // std::cout<<"# of invalid vertices: "<<invalidVertices.size()
  //   <<std::endl;

 /* 
  NODE_T invalid = 0;
  for(NODE_T i=0;i<n;i++) {
    if(colors[i] == -2)
     invalid++; 
  }
  std::cout<<invalid<<std::endl;
  */

  palStat[level].confColorTime = omp_get_wtime() - t1;
  
  // std::cout<<"Conflict Coloring Time: "<<confColorTime<<std::endl;

  nColors = *std::max_element(colors.begin(),colors.end()) + 1;

}

// double mem_access_time = 0;
// NODE_T sacrificial_counter = 0;

void fixBucketsCSR(NODE_T vtx, NODE_T col, NODE_T &vtxProcessed, 
    std::vector< std::vector<NODE_T> > &verBucket, std::vector<NODE_T> &verLoc, NODE_T &vMin) {
  for(auto v_id = h_confOffsets[vtx]; v_id < h_confOffsets[vtx+1]; v_id++) {
    auto v = h_confVertices[v_id];
    if(colors[v] == -1) {
      auto it = std::find(colList[v].begin(),colList[v].end(),col);
      if(it != colList[v].end()) {
        //fix the location of the last vertex 
        verLoc[verBucket[colList[v].size()-1].back()] = verLoc[v];
        //now swap v with last element
        std::swap(verBucket[colList[v].size()-1][verLoc[v]],verBucket[colList[v].size()-1].back());
        //remove v from this bucket
        verBucket[colList[v].size()-1].pop_back();
        
        //swap the colr with last color
        std::swap(*it,colList[v].back());
        //remove the color
        colList[v].pop_back(); 
        if(colList[v].empty() == true) {
          vtxProcessed++;
          invalidVertices.push_back(v);
          colors[v] = -2;
        }
        else {
          if (colList[v].size() < vMin) 
            vMin = colList[v].size();

          verBucket[colList[v].size()-1].push_back(v); 
          verLoc[v] = verBucket[colList[v].size()-1].size()-1;
        }
      }
      
    }
  }

  
}

void confColorRandCSR(std::vector<NODE_T> &nodeList) {
  
  //std::cout<<"conflicting edges: "<<nConflicts<<std::endl; 
  double t1 = omp_get_wtime();
  orderConfVerticesRand(nodeList);
  for(NODE_T i:nodeList) {
    if(h_confOffsets[i+1] - h_confOffsets[i] != 0) {
      for(auto col:colList[i]) {
        bool flag = true;
        for(auto v_id = h_confOffsets[i]; v_id < h_confOffsets[i+1]; v_id++) {
          auto v = h_confVertices[v_id];
          if (colors[v] == col) {
            flag = false;
            break; 
          }
        }     

        if(flag == true) {
          colors[i] = col;
          break; 
        }
      }  
      if(colors[i] == -1) {
        invalidVertices.push_back(i);
        colors[i] = -2;
      }
    } 
    else
      colors[i] = colList[i][0];
    //std::cout<<i<<" "<<colors[i]<<std::endl;
  }
  palStat[level].confColorTime = omp_get_wtime() - t1;
  //std::cout<<"# of vertices can not be colored: "<<invalidVertices.size()
   // <<std::endl;
  nColors = *std::max_element(colors.begin(),colors.end()) + 1;
}

void confColorRandCSR() {
  
  //std::cout<<"conflicting edges: "<<nConflicts<<std::endl; 
  double t1 = omp_get_wtime();
  orderConfVerticesRand();
  for(NODE_T i:vertexOrder) {
    if(h_confOffsets[i+1] - h_confOffsets[i] != 0) {
      for(auto col:colList[i]) {
        bool flag = true;
        for(auto v_id = h_confOffsets[i]; v_id < h_confOffsets[i+1]; v_id++) {
          auto v = h_confVertices[v_id];
          if (colors[v] == col) {
            flag = false;
            break; 
          }
        }     

        if(flag == true) {
          colors[i] = col;
          break; 
        }
      }  
      if(colors[i] == -1) {
        invalidVertices.push_back(i);
        colors[i] = -2;
      }
    } 
    else
      colors[i] = colList[i][0];
    //std::cout<<i<<" "<<colors[i]<<std::endl;
  }
  palStat[level].confColorTime = omp_get_wtime() - t1;
  //std::cout<<"# of vertices can not be colored: "<<invalidVertices.size()
   // <<std::endl;
  nColors = *std::max_element(colors.begin(),colors.end()) + 1;
}
#endif // ENABLE_GPU

//This function sort the vertices of the conflict graph w.r.t to their degrees.
//The idea is to color the vertex with highest degree first, since this represents
//the most conflicted with other vertices. Does not perform that well in practice.
//We are not using it at the moment.
void orderConfVerticesLF() {
   
  std::vector<NODE_T > degreeConf(n,0);
  vertexOrder.resize(n);

  for(NODE_T i=0;i<n;i++) {
    degreeConf[i]= confAdjList[i].size(); 
  }
  std::iota(vertexOrder.begin(),vertexOrder.end(),0);

  std::sort(vertexOrder.begin(),vertexOrder.end(),
      [&degreeConf] (NODE_T t1, NODE_T t2) {return degreeConf[t1] > degreeConf[t2];});


}

void orderConfVerticesRand() {

  vertexOrder.resize(n);
  std::iota(vertexOrder.begin(),vertexOrder.end(),0);
  std::mt19937 engine(seed);
  std::shuffle(vertexOrder.begin(),vertexOrder.end(),engine);
}

void orderConfVerticesRand(std::vector<NODE_T> &nodeList) {
  std::mt19937 engine(seed);
  std::shuffle(nodeList.begin(),nodeList.end(),engine);
}

//This function attempt to color the conflicting graph with largest degree heuristics.
void confColorLF() {
  
  //std::cout<<"conflicting edges: "<<nConflicts<<std::endl; 
  double t1 = omp_get_wtime();
  orderConfVerticesLF();
  for(NODE_T i:vertexOrder) {
    if(confAdjList[i].empty() == false) {
      for(auto col:colList[i]) {
        bool flag = true;
        for(auto v:confAdjList[i]) {
          if (colors[v] == col) {
            flag = false;
            break; 
          }
        }     

        if(flag == true) {
          colors[i] = col;
          break; 
        }
      }  
      if(colors[i] == -1) {
        invalidVertices.push_back(i);
        colors[i] = -2;
      }
    } 
    else
      colors[i] = colList[i][0];
    //std::cout<<i<<" "<<colors[i]<<std::endl;
  }
  palStat[level].confColorTime = omp_get_wtime() - t1;
  //std::cout<<"# of vertices can not be colored: "<<invalidVertices.size()
   // <<std::endl;
  nColors = *std::max_element(colors.begin(),colors.end()) + 1;
}

//random order 
void confColorRand() {
  
  //std::cout<<"conflicting edges: "<<nConflicts<<std::endl; 
  double t1 = omp_get_wtime();
  orderConfVerticesRand();
  for(NODE_T i:vertexOrder) {
    if(confAdjList[i].empty() == false) {
      for(auto col:colList[i]) {
        bool flag = true;
        for(auto v:confAdjList[i]) {
          if (colors[v] == col) {
            flag = false;
            break; 
          }
        }     

        if(flag == true) {
          colors[i] = col;
          break; 
        }
      }  
      if(colors[i] == -1) {
        invalidVertices.push_back(i);
        colors[i] = -2;
      }
    } 
    else
      colors[i] = colList[i][0];
    //std::cout<<i<<" "<<colors[i]<<std::endl;
  }
  palStat[level].confColorTime = omp_get_wtime() - t1;
  //std::cout<<"# of vertices can not be colored: "<<invalidVertices.size()
   // <<std::endl;
  nColors = *std::max_element(colors.begin(),colors.end()) + 1;
}

void confColorRand(std::vector<NODE_T> &nodeList) {
  
  //std::cout<<"conflicting edges: "<<nConflicts<<std::endl; 
  double t1 = omp_get_wtime();
  orderConfVerticesRand(nodeList);
  for(NODE_T i:nodeList) {
    if(confAdjList[i].empty() == false) {
      for(auto col:colList[i]) {
        bool flag = true;
        for(auto v:confAdjList[i]) {
          if (colors[v] == col) {
            flag = false;
            break; 
          }
        }     

        if(flag == true) {
          colors[i] = col;
          break; 
        }
      }  
      if(colors[i] == -1) {
        invalidVertices.push_back(i);
        colors[i] = -2;
      }
    } 
    else
      colors[i] = colList[i][0];
    //std::cout<<i<<" "<<colors[i]<<std::endl;
  }
  palStat[level].confColorTime = omp_get_wtime() - t1;
  //std::cout<<"# of vertices can not be colored: "<<invalidVertices.size()
   // <<std::endl;
  nColors = *std::max_element(colors.begin(),colors.end()) + 1;
}

void confColorGreedy() {
  
  double t1 = omp_get_wtime();
  // std::cout<<"# of conflicting edges: "<<nConflicts<<std::endl; 
  //Buckets of size T;
  std::vector<std::vector<NODE_T> > verBucket(T+1);
  //to stoe the position of vertex in the corresponding bucket
  std::vector<NODE_T> verLocation(n);
  
  NODE_T vMin = T; 

  std::mt19937 engine(seed);
  std::uniform_int_distribution<NODE_T> uniform_dist(0, colThreshold-1);

  //# of verties processed. For stopping condition later
  NODE_T vtxProcessed = 0;
  for(NODE_T i=0;i<n;i++) {
    //if this is a non-conflicting vertex we can color it arbitrarily from the 
    //list of colors.
    if(confAdjList[i].empty() == true) {
      //std::cout<<"coloring non-conflicting edge"<<std::endl;
      uniform_dist = std::uniform_int_distribution<NODE_T>(0, 
          colList[i].size()-1);
      colors[i] = colList[i][uniform_dist(engine)];
      vtxProcessed++;
      continue;
    }
    //otherwise place it in appropriate bucket
    NODE_T t = colList[i].size();
    verBucket[t-1].push_back(i); 
    verLocation[i] = verBucket[t-1].size()-1;
    
    //the minimum bucket length
    if(t < vMin) {
      vMin = t; 
    }
  }
 
  //keep processing until we see all the vertices.
  // double buckets_time = 0.0;
  while (vtxProcessed < n) {
    for(NODE_T i = vMin-1; i < T;i++) {
      //if this bucket has elements
      //select a random vertex and swap it with the last element
      //to efficiently remove this vertex from the bucket.
      //std::cout<<"Bucket: "<<i<<"size: "<<verBucket[i].size()<<std::endl;
      if(verBucket[i].empty() == false) {
        uniform_dist = std::uniform_int_distribution<NODE_T>(0, 
            verBucket[i].size()-1);
        NODE_T selectedVtxLoc = uniform_dist(engine);
        NODE_T selectedVtx = verBucket[i].at(selectedVtxLoc);

        //update the verLocation array of the last element
        verLocation[verBucket[i].back()] = selectedVtxLoc;
        
        //swap the vertex with the last element
        std::swap(verBucket[i][selectedVtxLoc],verBucket[i].back());
        //remove the vertex
        verBucket[i].pop_back();

        //attempt to color this vertex
        NODE_T colInd = attemptToColor(selectedVtx);
        if(colInd>=0) {
          //remove col at colInd of the vtx is not necessary
          //std::swap(colList[selectedVtx][colInd],colList[selectedVtx].back());
          //colList[selectedVtx].pop_back();

          // double bucket_time = omp_get_wtime();
          fixBuckets(selectedVtx,colors[selectedVtx],vtxProcessed,verBucket,verLocation,vMin); 
          // buckets_time += omp_get_wtime() - bucket_time;
          //std::cout<<"updated VMin: "<<vMin<<std::endl;
        }
        else {
          colors[selectedVtx] = -2;
          invalidVertices.push_back(selectedVtx); 
        }
        vtxProcessed++;
        break;
      } 
    }
  }
  // std::cout << "Buckets time: " << buckets_time << std::endl;
  // std::cout<<"# of invalid vertices: "<<invalidVertices.size()
  //   <<std::endl;

 /* 
  NODE_T invalid = 0;
  for(NODE_T i=0;i<n;i++) {
    if(colors[i] == -2)
     invalid++; 
  }
  std::cout<<invalid<<std::endl;
  */

  palStat[level].confColorTime = omp_get_wtime() - t1;
  
  //std::cout<<"Conflict Coloring Time: "<<palStat[level].confColorTime<<std::endl;

  nColors = *std::max_element(colors.begin(),colors.end()) + 1;

}

void confColorGreedy(std::vector<NODE_T> &nodeList) {
  
  double t1 = omp_get_wtime();
  //std::cout<<"# of conflicting edges: "<<nConflicts<<std::endl; 
  //Buckets of size T;
  std::vector<std::vector<NODE_T> > verBucket(T+1);
  //to stoe the position of vertex in the corresponding bucket
  std::vector<NODE_T> verLocation(n);
  
  NODE_T vMin = T; 

  std::mt19937 engine(seed);
  std::uniform_int_distribution<NODE_T> uniform_dist(0, colThreshold-1);

  //# of verties processed. For stopping condition later
  NODE_T vtxProcessed = 0;
  for(NODE_T i:nodeList) {
    //if this is a non-conflicting vertex we can color it arbitrarily from the 
    //list of colors.
    if(confAdjList[i].empty() == true) {
      //std::cout<<"coloring non-conflicting edge"<<std::endl;
      uniform_dist = std::uniform_int_distribution<NODE_T>(0, 
          colList[i].size()-1);
      colors[i] = colList[i][uniform_dist(engine)];
      vtxProcessed++;
      continue;
    }
    //otherwise place it in appropriate bucket
    NODE_T t = colList[i].size();
    verBucket[t-1].push_back(i); 
    verLocation[i] = verBucket[t-1].size()-1;
    
    //the minimum bucket length
    if(t < vMin) {
      vMin = t; 
    }
  }
 
  //keep processing until we see all the vertices. 
  // double buckets_time = 0.0;
  while (vtxProcessed < nodeList.size()) {
    for(NODE_T i = vMin-1; i < T;i++) {
      //if this bucket has elements
      //select a random vertex and swap it with the last element
      //to efficiently remove this vertex from the bucket.
      //std::cout<<"Bucket: "<<i<<"size: "<<verBucket[i].size()<<std::endl;
      if(verBucket[i].empty() == false) {
        uniform_dist = std::uniform_int_distribution<NODE_T>(0, 
            verBucket[i].size()-1);
        NODE_T selectedVtxLoc = uniform_dist(engine);
        NODE_T selectedVtx = verBucket[i].at(selectedVtxLoc);

        //update the verLocation array of the last element
        verLocation[verBucket[i].back()] = selectedVtxLoc;
        
        //swap the vertex with the last element
        std::swap(verBucket[i][selectedVtxLoc],verBucket[i].back());
        //remove the vertex
        verBucket[i].pop_back();

        //attempt to color this vertex
        NODE_T colInd = attemptToColor(selectedVtx);
        if(colInd>=0) {
          //remove col at colInd of the vtx is not necessary
          //std::swap(colList[selectedVtx][colInd],colList[selectedVtx].back());
          //colList[selectedVtx].pop_back();

          // double bucket_time = omp_get_wtime();
          fixBuckets(selectedVtx,colors[selectedVtx],vtxProcessed,verBucket,verLocation,vMin); 
          // buckets_time += omp_get_wtime() - bucket_time;
          //std::cout<<"updated VMin: "<<vMin<<std::endl;
        }
        else {
          colors[selectedVtx] = -2;
          invalidVertices.push_back(selectedVtx); 
        }
        vtxProcessed++;
        break;
      } 
    }
  }
  // std::cout << "Buckets time: " << buckets_time << std::endl;
 // std::cout<<"# of invalid vertices: "<<invalidVertices.size()
   // <<std::endl;

 /* 
  NODE_T invalid = 0;
  for(NODE_T i=0;i<n;i++) {
    if(colors[i] == -2)
     invalid++; 
  }
  std::cout<<invalid<<std::endl;
  */

  palStat[level].confColorTime = omp_get_wtime() - t1;
  
  //std::cout<<"Conflict Coloring Time: "<<confColorTime<<std::endl;

  nColors = *std::max_element(colors.begin(),colors.end()) + 1;

}

//initialize for the recursive implementation. 
  void reInit(std::vector<NODE_T> & nodeList,NODE_T colThresh, float alpha=1, NODE_T lst_sz = -1) {
    std::sort(nodeList.begin(), nodeList.end());
    colThreshold = colThresh; 
    nConflicts = 0;
    //The nodes in the node List need to have color -1.
    for(auto u:nodeList) {
      colors[u] = -1; 
      colList[u].clear();
      confAdjList[u].clear();
    }
    //for memory improvement
    for (auto i=0;i<colList.size();i++) {
      colList[i].clear();
      confAdjList[i].clear(); 
    }
    #ifdef ENABLE_GPU
    h_colList.clear();
    // h_confOffsets.clear();
    // h_confVertices.clear();
    // Free d_colList
    if (d_colList != nullptr) {
      ERR_CHK(cudaFree(d_colList));
      d_colList = nullptr;
    }
    // if (d_confOffsets != nullptr) {
    //   ERR_CHK(cudaFree(d_confOffsets));
    //   d_confOffsets = nullptr;
    // }
    if (d_confVertices != nullptr) {
      ERR_CHK(cudaFree(d_confVertices));
      d_confVertices = nullptr;
    }
    #endif // ENABLE_GPU
    if(lst_sz < 0)
      T =  static_cast<NODE_T> (alpha*log(nodeList.size()));
    else
      T = lst_sz;

    if(T>colThresh)
      T = colThresh;
    invalidVertices.clear();

    palStat.push_back({(NODE_T)nodeList.size(),-1,-1,colThreshold,T,0,0.0,0.0,0.0});
    level = level + 1;
    assignListColor(nodeList,getNumColors());
  }
  
  template<typename PauliTy = std::string>
  bool checkValidity( ClqPart::JsonGraph &jsongraph) {
  for(NODE_T eu =0; eu < n-1; eu++) {
    for(NODE_T ev = eu+1; ev < n; ev++) {

      if(jsongraph.is_an_edge<PauliTy>(eu,ev) == false) {
        if((colors[eu] >  -1 && colors[ev] > -1) && (colors[eu] == colors[ev])) 
          return false;
      }
    }
  }
  return true;
}

  std::vector< std::vector<NODE_T> >& getConfAdjList() { return confAdjList; }
  //std::vector<NODE_T>& getConfVertices() { return confVertices; }
  std::vector<NODE_T>& getInvVertices() { return invalidVertices; }
  std::vector<NODE_T> getColors() { return colors; }
  NODE_T getNumColors() {return nColors;}
  EDGE_T getNumConflicts() {return nConflicts;}
  PalColStat getPalStat(int lev=0) {return palStat[lev];}

private:
//This function assign random list of colors from the Palette.
void assignListColor() {

  std::mt19937 engine(seed);
  std::uniform_int_distribution<NODE_T> uniform_dist(0, colThreshold-1);
  
  //We want the colors to be not repeating. Thus initially the list size is
  //same for each vertex
  std::vector<bool> isPresent(colThreshold);

  // std::cout<<"Assigning random list color " << "Palette Size: "<<colThreshold
  //           <<" "<<"List size: "<<T<<std::endl;
  
  double t1 = omp_get_wtime();
  #ifdef ENABLE_GPU
  h_colList.reserve(n * T);
  #endif // ENABLE_GPU
  for (NODE_T i= 0; i<n ; i++) {
    std::fill(isPresent.begin(),isPresent.end(),false);
    for (NODE_T j=0; j<T ; j++) {
      NODE_T col;
      do {
        col = uniform_dist(engine);
      }while(isPresent[col] == true);

      colList[i].push_back(col); 
      isPresent[col] = true;
    } 
    std::sort(colList[i].begin(),colList[i].end());
    #ifdef ENABLE_GPU
    h_colList.insert(h_colList.end(), colList[i].begin(), colList[i].end());
    #endif
  }
  #ifdef ENABLE_GPU
  // Copy h_colList to GPU
  cudaError_t err;
  ERR_CHK(cudaMalloc(&d_colList, h_colList.size() * sizeof(NODE_T)));
  ERR_CHK(cudaMemcpy(d_colList, h_colList.data(), h_colList.size() * sizeof(NODE_T), cudaMemcpyHostToDevice));
  #endif // ENABLE_GPU
  palStat[level].assignTime = omp_get_wtime() - t1;
  // std::cout<<"Assignment Time: "<<assignTime<<std::endl;

}

//overloaded function to work with subset of nodes
void assignListColor(std::vector<NODE_T> &nodeList,NODE_T offset) {

  std::mt19937 engine(seed);
  std::uniform_int_distribution<NODE_T> uniform_dist(offset, offset+colThreshold-1);
  
  //We want the colors to be not repeating. Thus initially the list size is
  //same for each vertex
  std::vector<bool> isPresent(colThreshold);

  //std::cout<<"Assigning random list color " << "Palette Size: "<<colThreshold
   //         <<" "<<"List size: "<<T<<std::endl;
  
  double t1 = omp_get_wtime();
  #ifdef ENABLE_GPU
  h_colList.reserve(nodeList.size() * T);
  #endif // ENABLE_GPU
  for (NODE_T i:nodeList) {
    std::fill(isPresent.begin(),isPresent.end(),false);
    //clear the colList since it may contain color from previous recursive level. 
    colList[i].clear();
    for (NODE_T j=0; j<T ; j++) {
      NODE_T col;
      do {
        col = uniform_dist(engine);
      }while(isPresent[col-offset] == true);

      colList[i].push_back(col); 
      isPresent[col-offset] = true;
    } 
    std::sort(colList[i].begin(),colList[i].end());
    #ifdef ENABLE_GPU
    h_colList.insert(h_colList.end(), colList[i].begin(), colList[i].end());
    #endif
  }
  #ifdef ENABLE_GPU
  // Copy h_colList to GPU
  ERR_CHK(cudaMalloc(&d_colList, h_colList.size() * sizeof(NODE_T)));
  ERR_CHK(cudaMemcpy(d_colList, h_colList.data(), h_colList.size() * sizeof(NODE_T), cudaMemcpyHostToDevice));
  #endif // ENABLE_GPU
  palStat[level].assignTime = omp_get_wtime() - t1;
  //std::cout<<"Assignment Time: "<<assignTime<<std::endl;
}

  /********************************************************************************
//The next three functions are not necessary at this moment. They are used to color
//the invalid vertices. But since we do not know the subgraph induced by the invalid
//vertices coloring them only with conflict graph is wrong.  
void populateCandColors( NODE_T u, std::vector<NODE_T> &candColors) {
  for(auto v:confAdjList[u] ) {
    if (confColors[v] !=  -1) {
      candColors[ confColors[v] ] = u; 
    } 
  }
}

NODE_T firstAvailColor( NODE_T u, std::vector<NODE_T> &candColors) {
  for(NODE_T v=0;v < candColors.size(); v++ ) {
    if ( candColors[v] != u) {
      return v; 
    } 
  }
  return -1;
}
void greedyColor(NODE_T offset) {
  std::vector<NODE_T> candColors(n,-1);
 
  for(auto u:invalidVertices ) {
    //std::cout<<u<<std::endl;
    populateCandColors(u,candColors); 
    
    confColors[u] = firstAvailColor(u,candColors);
    colors[u] = confColors[u]+offset;
  }
}
********************************************************************************/
  //The function given a vtx, properly color the vertx using a random color from the list.
//the coloring is guaranteed but I first name it attempt. kept in this way.
NODE_T attemptToColor(NODE_T vtx) {

  std::mt19937 engine(seed);
  std::uniform_int_distribution<NODE_T> uniform_dist(0,colList[vtx].size()-1);

  auto colInd = uniform_dist(engine);
  
  auto col = colList[vtx].at(colInd);
  colors[vtx] = col;
  return colInd;
}
  //The next two functions fixBuckets and ConfColorGreedy are needed for coloring
//the conflict graph. This heuristic color the vertices with the smallest list size
//first, since they have less options. Works well in practice. 
void fixBuckets(NODE_T vtx, NODE_T col, NODE_T &vtxProcessed, 
    std::vector< std::vector<NODE_T> > &verBucket, std::vector<NODE_T> &verLoc, NODE_T &vMin) {
  for( auto v:confAdjList[vtx] ) {
    if(colors[v] == -1) {
      auto it = std::find(colList[v].begin(),colList[v].end(),col);
      if(it != colList[v].end()) {
        //fix the location of the last vertex 
        verLoc[verBucket[colList[v].size()-1].back()] = verLoc[v];
        //now swap v with last element
        std::swap(verBucket[colList[v].size()-1][verLoc[v]],verBucket[colList[v].size()-1].back());
        //remove v from this bucket
        verBucket[colList[v].size()-1].pop_back();
        
        //swap the colr with last color
        std::swap(*it,colList[v].back());
        //remove the color
        colList[v].pop_back(); 
        if(colList[v].empty() == true) {
          vtxProcessed++;
          invalidVertices.push_back(v);
          colors[v] = -2;
        }
        else {
          if (colList[v].size() < vMin) 
            vMin = colList[v].size();

          verBucket[colList[v].size()-1].push_back(v); 
          verLoc[v] = verBucket[colList[v].size()-1].size()-1;
        }
      }
      
    }
  }  
}
};

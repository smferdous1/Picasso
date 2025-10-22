/*
 * Integrates ECLgraph (.egr) CSR input with PaletteColor.
 */

#include <iostream>
#include <cstring>
#include <fstream>

#include "ECLgraph.h"
#include "ClqPart/paletteCol.h"
#include "cxxopts/cxxopts.hpp"

static void printStat(int level, PalColStat &palStat) {
  std::cout << "***********Level " << level << "*******" << std::endl;
  std::cout << "Num Nodes: " << palStat.n << "\n";
  std::cout << "Num Edges: " << palStat.m << "\n";
  std::cout << "Avg. Deg.: " << (double)2 * palStat.m / palStat.n << "\n";
  std::cout << "Palette Size: " << palStat.palSz << "\n";
  std::cout << "List Size: " << palStat.lstSz << "\n";
  std::cout << "Num Conflict Edges: " << palStat.mConf << "\n";
  std::cout << "Conflict to Edge (%): " << (double)palStat.mConf / palStat.m * 100 << "\n";
  std::cout << "Num Colors: " << palStat.nColors << std::endl;
  std::cout << "Assign Time: " << palStat.assignTime << "\n";
  std::cout << "Conf. Build Time: " << palStat.confBuildTime << "\n";
  std::cout << "Conf. Color Time: " << palStat.confColorTime << "\n\n";
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("palcolEcl", "Palette coloring reading ECL .egr graphs (CSR)");
  options.add_options()
    ("in,infile", "input .egr file", cxxopts::value<std::string>())
    ("t,target", "target palette size (>=1)", cxxopts::value<NODE_T>())
    ("a,alpha", "coefficient to log(n) for list size", cxxopts::value<float>()->default_value("1.0"))
    ("l,list", "explicit list size (overrides alpha)", cxxopts::value<NODE_T>()->default_value("-1"))
    ("o,order", "RANDOM or LIST (unused for LF)", cxxopts::value<std::string>()->default_value("LIST"))
    ("c,check", "check validity (not applicable without original graph semantics)", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "print usage");

  std::string inFname, orderName;
  NODE_T target, list_size;
  float alpha;
  bool isValid;
  try {
    auto result = options.parse(argc, argv);
    if (result.count("help")) {
      std::cout << options.help() << "\n";
      return 0;
    }
    inFname = result["infile"].as<std::string>();
    target = result["target"].as<NODE_T>();
    alpha = result["alpha"].as<float>();
    list_size = result["list"].as<NODE_T>();
    orderName = result["order"].as<std::string>();
    isValid = result["check"].as<bool>();
  } catch (cxxopts::exceptions::exception&) {
    std::cout << options.help() << std::endl;
    return 1;
  }

  // Read ECL graph
  ECLgraph g = readECLgraph(inFname.c_str());
  const NODE_T n = static_cast<NODE_T>(g.nodes);

  if (list_size >= 0) {
    std::cout << "Since list size is given, ignoring alpha" << std::endl;
  }

  PaletteColor<> palcol(n, target, alpha, list_size);

  double t1 = omp_get_wtime();
  // Treat input edges as complement-graph edges, as in appPaletteColor.cc
  for (int i = 0; i < g.nodes; i++) {
    const int beg = g.nindex[i];
    const int end = g.nindex[i + 1];
    for (int e = beg; e < end; e++) {
      const int j = g.nlist[e];
      if (i < j) {
        palcol.buildStreamConfGraph(static_cast<NODE_T>(i), static_cast<NODE_T>(j));
      }
    }
  }
  double createConfTime = omp_get_wtime() - t1;

  // Color conflict graph using LF heuristic (matches CPU app)
  palcol.confColorLF();

  PalColStat palStat = palcol.getPalStat();
  palStat.m = static_cast<EDGE_T>(g.edges);
  palStat.confBuildTime = createConfTime;
  palStat.nColors = palcol.getNumColors();
  printStat(0, palStat);

  std::vector<NODE_T> invVert = palcol.getInvVertices();
  std::cout << "# of Invalid Vertices: " << invVert.size() << "\n";
  if (isValid) {
    std::cout << "Validity check is not supported for generic ECL graphs." << std::endl;
  }

  freeECLgraph(g);
  return 0;
}


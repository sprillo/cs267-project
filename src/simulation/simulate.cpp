/**
 * @file simulate.cpp
 * @author Xingyu Li (xingyuli9961@berkeley.edu)
 * @brief This is the file contains the C++ implementation of the simulation function
 * @version 0.1
 * @date 2022-04-20
 * 
 * @copyright Copyright (c) 2022
 * 
 * Prerequisite:
 * module load openmpi
 * Compile the code:
 * mpicxx -o simulate simulate.cpp
 * An example testing command use for developing and checking if the file compiles:
 * ./simulate ./../../tests/simulation_tests/test_input_data/tree_dir
 * 
 * Below shows the testing arguments during development:
 * (This is a sample version of the test_simulate_msas_normal_model)
 * argv[1]  (tree_dir): "./../../tests/simulation_tests/test_input_data/tree_dir"
 * argv[2]  (site_rates_dir): "./../../tests/simulation_tests/test_input_data/synthetic_site_rates_dir"
 * argv[3]  (contact_map_dir): "./../../tests/simulation_tests/test_input_data/synthetic_contact_map_dir"
 * argv[4]  (num_of_families): 3
 * argv[5]  (num_of_amino_acids): 2
 * argv[6]  (pi_1_path): "./../../tests/simulation_tests/test_input_data/normal_model/pi_1.txt"
 * argv[7]  (Q_1_path): "./../../tests/simulation_tests/test_input_data/normal_model/Q_1.txt"
 * agrv[8]  (pi_2_path): "./../../tests/simulation_tests/test_input_data/normal_model/pi_2.txt"
 * argv[9]  (Q_2_path): "./../../tests/simulation_tests/test_input_data/normal_model/Q_2.txt"
 * argv[10] (strategy): "all_transitions"
 * argv[11] (output_msa_dir): "./../../tests/simulation_tests/test_input_data/simulated_msa_dir"
 * argv[12] (random_seed): 0
 * argv[13] (num_processes): 3
 * argv[14 : 17] (families): ["fam1", "fam2", "fam3"]
 * argv[17 : 19] (amino_acids): ["S", "T"]
 * 
 */
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <mpi.h>

int main(int argc, char *argv[]) {
    std::cout << "This is the start of this testing file ..." << std::endl;
    std::string tree_dir = argv[1];
    std::string site_rates_dir = argv[2];
    std::string contact_map_dir = argv[3];
    int num_family = argv[4]
    std::cout << tree_dir << std::endl;
}
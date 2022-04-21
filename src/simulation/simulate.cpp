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
 * An example testing command use for developing and checking if the file compiles can be found in the test_simulate.sh file.
 * 
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
 * argv[13 : 16] (families): ["fam1", "fam2", "fam3"]
 * argv[16 : 18] (amino_acids): ["S", "T"]
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

// Helper function to read the tree
void read_tree(std::string tree_dir, std::string family) {

}



// Initialize simulation on each process
void init_simulation() {

}

// Run simulation for all the families assigned to a certain process
void run_simulation(std::vector<std::string> families) {
    // Iterate through all the families allocated:
    for (std::string s : families) {
        std::cout << "The current family is " << s << std::endl;

    }
}


int main(int argc, char *argv[]) {
    // Start execution
    std::cout << "This is the start of this testing file ..." << std::endl;
    
    // Init MPI
    // int num_procs, rank;
    // MPI_Init(&argc, &argv);
    // MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Print MPI parameters
    // std::cout << "The number of process is " << num_procs << std::endl;
    // std::cout << "The current rank is " << rank << std::endl;
    
    // Read in all the arguments
    std::string tree_dir = argv[1];
    std::string site_rates_dir = argv[2];
    std::string contact_map_dir = argv[3];
    int num_of_families = atoi(argv[4]);
    int num_of_amino_acids = atoi(argv[5]);
    std::string pi_1_path = argv[6];
    std::string Q_1_path = argv[7];
    std::string pi_2_path = argv[8];
    std::string Q_2_path = argv[9];
    std::string strategy = argv[10];
    std::string output_msa_dir = argv[11];
    int random_seed = atoi(argv[12]);
    std::vector<std::string> families;
    families.reserve(num_of_families);
    for (int i = 0; i < num_of_families; i++) {
        families.push_back(argv[13 + i]);
    }
    std::vector<std::string> amino_acids;
    amino_acids.reserve(num_of_amino_acids);
    for (int i = 0; i < num_of_amino_acids; i++) {
        amino_acids.push_back(argv[13 + num_of_families + i]);
    }

    // Below is just for testing the proper arg parsing
    // std::cout << "Reading arguments ..." << std::endl;
    // std::cout << "The tree_dir is " << tree_dir << std::endl;
    // std::cout << "The site_rates_dir is " << site_rates_dir << std::endl;
    // std::cout << "The contact_map_dir is " << contact_map_dir << std::endl;
    // std::cout << "The number of families is " << num_of_families << " and they are: " << std::endl;
    // for (std::string s : families) {
    //     std::cout << s << std::endl;
    // }
    // std::cout << "The number of amino acids is " << num_of_amino_acids << " and they are: " << std::endl;
    // for (std::string s : amino_acids) {
    //     std::cout << s << std::endl;
    // }
    // std::cout << "The pi_1_path is " << pi_1_path << std::endl;
    // std::cout << "The Q_1_path is " << Q_1_path << std::endl;
    // std::cout << "The pi_2_path is " << pi_2_path << std::endl;
    // std::cout << "The Q_2_path is " << Q_2_path << std::endl;
    // std::cout << "The strategy is " << strategy << std::endl;
    // std::cout << "The output_msa_dir is " << output_msa_dir << std::endl;
    // std::cout << "The strategy is " << strategy << std::endl;
    // std::cout << "The random_seed is " << random_seed << std::endl;

    // Initialize simulation
    init_simulation();

    // Run the simulation
    run_simulation(families);


    // MPI_Finalize();
}
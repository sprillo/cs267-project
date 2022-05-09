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
 * argv[19] (load_balancing_mode): 0 (0: naive version; 1: zig-zag)
 * argv[20] (familiy_file_path): ./test_familiy_sizes.txt (If load_balancing_mode == 1)
 * 
 */
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <random>
#include <vector>
#include <unordered_map>
#include <set>
#include <map>
#include <algorithm>
#include <assert.h>
#include <pthread.h>

#include <omp.h>
#include <mpi.h>

// global variables
std::vector<float> p1_probability_distribution;
std::vector<float> p2_probability_distribution;
std::vector<std::vector<float>> Q1_rate_matrix;
std::vector<std::vector<float>> Q2_rate_matrix;
std::vector<std::string> amino_acids_alphabet;
std::vector<std::string> amino_acids_pairs;
std::default_random_engine* random_engines;
std::discrete_distribution<int>* Q1_CTPs;
std::discrete_distribution<int>* Q2_CTPs;
std::vector<pthread_mutex_t> Q1_CTPs_mutex;
std::vector<pthread_mutex_t> Q2_CTPs_mutex;

#define DEBUG 0

// Adjacent value pairs
typedef struct {
    /* data */
    std::string node;
    float length;
} adj_pair_t;

// The tree class
class Tree {
    public:

    int num_of_nodes;
    std::unordered_map<std::string, std::vector<adj_pair_t>> adjacent_list;
    int m;
    std::unordered_map<std::string, int> out_deg;
    std::unordered_map<std::string, int> in_deg;
    std::unordered_map<std::string, adj_pair_t> parent_map;

    Tree(int num_nodes) {
        num_of_nodes = num_nodes;
        adjacent_list.clear();
        m = 0;
        out_deg.clear();
        in_deg.clear();
        parent_map.clear();
    }

    void add_node(std::string v) {
        std::vector<adj_pair_t> v_list;
        adjacent_list[v] = v_list;
        out_deg[v] = 0;
        in_deg[v] = 0;
    }

    void add_edge(std::string u, std::string v, float length) {
        adj_pair_t adjacent_pair;
        adjacent_pair.node = v;
        adjacent_pair.length = length;
        adjacent_list[u].push_back(adjacent_pair);
        m += 1;
        out_deg[u] += 1;
        in_deg[v] += 1;
        if (parent_map.find(v) != parent_map.end()) {
            std::cerr << "Node " << v << " already has a parent, graph is not a tree." << std::endl;
            exit(1);
        }
        adj_pair_t parent_pair;
        parent_pair.node = u;
        parent_pair.length = length;
        parent_map[v] = parent_pair;
    }

    bool is_node(std::string v) {
        if (adjacent_list.find(v) != adjacent_list.end()) {
            return true;
        } else {
            return false;
        }
    }

    std::vector<std::string> nodes() {
        std::vector<std::string> nodes_vector;
        for (auto const& kv : adjacent_list) {
            nodes_vector.push_back(kv.first);
        }
        return nodes_vector;
    }

    std::string root() {
        std::vector<std::string> roots;
        for (auto const& kv : adjacent_list) {
            if (in_deg[kv.first] == 0) {
                roots.push_back(kv.first);
            }
        }
        if (roots.size() != 1) {
            std::cerr << "There should be only 1 root, but there is/are " << roots.size() << " root(s)." << std::endl;
            exit(1);
        }
        return roots[0];
    }
    
    std::vector<adj_pair_t> children(std::string u) {
        return adjacent_list[u];
    }

    bool is_leaf(std::string u) {
        if (out_deg[u] == 0) {
            return true;
        } else {
            return false;
        }
    }

    void dfs (std::vector<std::string>& result, std::string v) {
        result.push_back(v);
        for (auto kv : children(v)) {
            dfs(result, kv.node);
        }
    }

    std::vector<std::string> preorder_traversal() {
        std::vector<std::string> result;
        dfs(result, root());
        return result;
    }

    adj_pair_t parent(std::string u) {
        return parent_map[u];
    }
};


// Helper function to read the tree
Tree read_tree(std::string treefilename, double* reading_tree_time, double* building_tree_time) {
    auto start_reading_tree = std::chrono::high_resolution_clock::now();
    int num_nodes;
    int num_edges;
    int edges_count = 0;
    std::string tmp;

    std::ifstream treefile;
    treefile.open(treefilename);

    treefile >> tmp;
    num_nodes = std::stoi(tmp);
    treefile >> tmp;
    if (tmp != "nodes") {
        std::cerr << "Tree file:" << treefilename << "should start with '[num_nodes] nodes'." << std::endl;
        exit(1);
    }
    // Tree newTree(num_nodes);
    std::vector<std::string> nodes(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        treefile >> tmp;
        nodes[i] = tmp;
        // newTree.add_node(tmp);
    }
    treefile >> tmp;
    num_edges = std::stoi(tmp);
    std::vector<std::string> parents(num_edges);
    std::vector<std::string> children(num_edges);
    std::vector<float> lengths(num_edges);
    treefile >> tmp;
    if (tmp != "edges") {
        std::cerr << "Tree file:" << treefilename << "should have line '[num_edges] edges' at position line " << num_nodes + 1 << std::endl;
        exit(1);
    }
    getline(treefile, tmp); // Get rid of the empty line left by reading the word
    while (treefile.peek() != EOF) {
        edges_count += 1;
        std::string u, v, l;
        float length;

        getline(treefile, tmp);
        std::stringstream tmpstring(tmp);
        tmpstring >> u;
        tmpstring >> v;
        tmpstring >> l;
        length = std::stof(l);
        // I didn't check the types at this point in the way python code does.
        // if (!newTree.is_node(u) || !newTree.is_node(v)) {
        //     std::cerr << "In Tree file " << treefilename << ": " << u << " and " << v << " should be nodes in the tree, but not." << std::endl;
        //     exit(1);
        // }
        // newTree.add_edge(u, v, length);
        parents[edges_count - 1] = u;
        children[edges_count - 1] = v;
        lengths[edges_count - 1] = length;
    }
    if (num_edges != edges_count) {
        std::cerr << "Tree file:" << treefilename << "should have " << num_edges << " edges, but it has " << edges_count << " instead." << std::endl;
        exit(1);
    }
    auto end_reading_tree = std::chrono::high_resolution_clock::now();

    auto start_building_tree = std::chrono::high_resolution_clock::now();
    Tree newTree(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        newTree.add_node(nodes[i]);
    }
    for (int i = 0; i < num_edges; i++){
        std::string u = parents[i];
        std::string v = children[i];
        float length = lengths[i];
        if (!newTree.is_node(u) || !newTree.is_node(v)) {
            std::cerr << "In Tree file " << treefilename << ": " << u << " and " << v << " should be nodes in the tree, but not." << std::endl;
            exit(1);
        }
        newTree.add_edge(u, v, length);
    }
    auto end_building_tree = std::chrono::high_resolution_clock::now();

    *reading_tree_time = std::chrono::duration<double>(end_reading_tree - start_reading_tree).count();
    *building_tree_time = std::chrono::duration<double>(end_building_tree - start_building_tree).count();

    return newTree;
}

// Read the site rates file
std::vector<float> read_site_rates(std::string filename) {
    int num_sites;
    std::string tmp;
    std::vector<float> result;

    std::ifstream siteratefile;
    siteratefile.open(filename);

    siteratefile >> tmp;
    num_sites = std::stoi(tmp);
    siteratefile >> tmp;
    if (tmp != "sites") {
        std::cerr << "Site rates file: " << filename << " should start with line '[num_sites] sites', but started with: " << tmp << " instead." << std::endl;
        exit(1);
    }
    while (siteratefile >> tmp) {
        result.push_back(std::stof(tmp));
    }
    if (int(result.size()) != num_sites) {
        std::cerr << "Site rates file: " << filename << " was supposed to have " << num_sites << " sites, but it has " << result.size() << "." << std::endl;
        exit(1);
    }
    return result;
}

// Read the contact map
std::vector<std::vector<int>> read_contact_map(std::string filename) {
    int num_sites;
    std::string tmp;
    std::vector<std::vector<int>> result;
    int line_count = 0;

    std::ifstream contactmapfile;
    contactmapfile.open(filename);

    contactmapfile >> tmp;
    num_sites = std::stoi(tmp);
    contactmapfile >> tmp;
    if (tmp != "sites") {
        std::cerr << "Contact map file: " << filename << " should start with line '[num_sites] sites', but started with: " << tmp << " instead." << std::endl;
        exit(1);
    }
    
    while (contactmapfile >> tmp) {
        line_count += 1;
        std::vector<int> row(num_sites, 0);
        for (int i = 0; i < num_sites; i++) {
            char a = tmp[i];
            if (a == '1') {
                row[i] = 1;
            } else if(a == '0'){
                row[i] = 0;
            } else{
                std::cerr << "Contact map file: " << filename << " should have only ones or zeros." << std::endl;
                exit(1);
            }
        }
        result.push_back(row);
    }
    if (num_sites != line_count) {
        std::cerr << "Contact map file: " << filename << " should have " << num_sites << " rows, but has " << line_count << "." << std::endl;
        exit(1);
    }
    return result;
}

// Read the probability distribution
std::vector<float> read_probability_distribution(std::string filename, const std::vector<std::string> & element_list) {
    std::vector<float> result;
    std::vector<std::string> states;
    std::string tmp, tmp2;
    float sum = 0;

    std::ifstream pfile;
    pfile.open(filename);

    getline(pfile, tmp);
    std::stringstream tmpstring(tmp);
    tmpstring >> tmp2;
    if (tmp2 != "state") {
        std::cerr << "Probability distribution file" << filename << "should have state here but have " << tmp << " instead." << std::endl;
        exit(1);
    }
    tmpstring >> tmp2;
    if (tmp2 != "prob") {
        std::cerr << "Probability distribution file" << filename << "should have prob here but have " << tmp << " instead." << std::endl;
        exit(1);
    }
    
    while(pfile.peek() != EOF) {
        std::string s;
        float p;

        getline(pfile, tmp);
        std::stringstream tmpstring2(tmp);
        tmpstring2 >> s;
        states.push_back(s);
        tmpstring2 >> tmp2;
        p = std::stof(tmp2);
        sum += p;
        result.push_back(p);
    }
    
    float diff = std::abs(sum - 1.0);
    if (diff > 0.000001) {
        std::cout << "Probability distribution at " << filename << " should add to 1.0, with a tolerance of 1e-6." << std::endl;
        exit(1);
    }

    if (states != element_list) {
        std::cerr << "Probability distribution file" << filename << " use a different (order of) alphabet." << std::endl;
        exit(1);
    }
    return result;
}

// Read the rate matrix
std::vector<std::vector<float>> read_rate_matrix(std::string filename, const std::vector<std::string> & element_list) {
    std::vector<std::vector<float>> result;
    std::vector<std::string> states1;
    std::vector<std::string> states2;
    std::string tmp, tmp2;

    std::ifstream qfile;
    qfile.open(filename);

    getline(qfile, tmp);
    std::stringstream tmpstring(tmp);
    while (tmpstring >> tmp2) {
        states1.push_back(tmp2);
    }
    
    while(qfile.peek() != EOF) {
        std::vector<float> row;
        getline(qfile, tmp);
        std::stringstream tmpstring2(tmp);
        tmpstring2 >> tmp2;
        states2.push_back(tmp2);
        while (tmpstring2 >> tmp2) {
            float p = std::stof(tmp2);
            row.push_back(p);
        }
        result.push_back(row);
    }

    if (states1 != element_list) {
        std::cerr << "Rate matrix file" << filename << " use a different (order of) alphabet." << std::endl;
        exit(1);
    }

    if (states2 != element_list) {
        std::cerr << "Rate matrix file" << filename << " use a different (order of) alphabet." << std::endl;
        exit(1);
    }

    return result;
}

// Read family sizes file and return the zig-zagged list of families.
std::vector<std::string> read_family_sizes(const std::vector<std::string> & families, std::string family_sizes_file, int load_balancing_mode, int num_procs) {
    std::vector<std::pair<int, std::string>> family_pairs;
    std::vector<std::string> result;
    std::string tmp, tmp1, tmp2, tmp3;

    std::ifstream famfile;
    famfile.open(family_sizes_file);

    getline(famfile, tmp);
    if (tmp != "family sequences sites") {
        std::cerr << "Family file" << family_sizes_file << " has a wrong format." << std::endl;
        exit(1);
    }

    std::set<std::string> families_all_set(families.begin(), families.end());

    while (famfile.peek() != EOF) {
        getline(famfile, tmp);
        std::stringstream tmpstring(tmp);
        tmpstring >> tmp1;
        tmpstring >> tmp2;
        tmpstring >> tmp3;
        if(families_all_set.count(tmp1))
            family_pairs.push_back(std::make_pair(std::stoi(tmp2) * std::stoi(tmp3), tmp1));
    }
    if(family_pairs.size() != families.size()){
        std::cerr << "Some family is missing in the family_sizes_file " << family_sizes_file << std::endl;
        exit(1);
    }
    if (load_balancing_mode == 0) {
        for (auto p : family_pairs) {
            result.push_back(p.second);
        }
    } else if (load_balancing_mode == 1) {
        sort(family_pairs.rbegin(), family_pairs.rend());
        for (int i = 0; i < 2 * num_procs * std::floor(family_pairs.size() / (2 * num_procs)); i += 2 * num_procs) {
            for (int j = 0; j < num_procs; j += 1) {
                result.push_back(family_pairs[i + j].second);
            }
            for (int j = 0; j < num_procs; j += 1) {
                result.push_back(family_pairs[i + 2 * num_procs - 1  - j].second);
            }
        }
        for (int i = 2 * num_procs * std::floor(family_pairs.size() / (2 * num_procs)); i < int(family_pairs.size()); i += 1) {
            result.push_back(family_pairs[i].second);
        }
    }

    return result;
}

// Write msa files
void write_msa(std::string filename, const std::map<std::string, std::vector<char>> & msa) {
    std::ofstream outfile;
    outfile.open(filename);

    for (auto key_value : msa) {
        outfile << ">" << key_value.first << std::endl;
        std::string tmp(key_value.second.begin(), key_value.second.end());
        outfile << tmp << std::endl;
    }

    outfile.close();
}

// Sample root state
std::vector<int> sample_root_states(int num_independent_sites, int num_contacting_pairs) {
    std::vector<int> result(num_independent_sites + num_contacting_pairs, 0);

    // First sample the independent sites
    std::discrete_distribution distribution1(cbegin(p1_probability_distribution), cend(p1_probability_distribution));
    for (int i = 0; i < num_independent_sites; i++) {
        result[i] = distribution1(random_engines[i]);
    }

    // Then sample the contacting sites
    std::discrete_distribution distribution2(cbegin(p2_probability_distribution), cend(p2_probability_distribution));
    for (int j = 0; j < num_contacting_pairs; j++) {
        result[num_independent_sites + j] = distribution2(random_engines[num_independent_sites + j]);
    }

    return result;
}

// Sample a transition
int sample_transition(int index, int starting_state, float elapsed_time, bool if_independent) {
    int current_state = starting_state;
    float current_time = 0;
    while (true) {
        float current_rate;
        if (if_independent) {
            current_rate = - Q1_rate_matrix[current_state][current_state];
        } else {
            current_rate = - Q2_rate_matrix[current_state][current_state];
        }
        // See when the next transition happens
        std::exponential_distribution<float> distribution1(current_rate);
        float waiting_time = distribution1(random_engines[index]);
        current_time += waiting_time;
        if (current_time >= elapsed_time) {
            // We reached the end of the process
            return current_state;
        }
        // Update the current_state;
        int new_state = -1;
        if (if_independent) {
            pthread_mutex_lock(&Q1_CTPs_mutex[current_state]);
            new_state = Q1_CTPs[current_state](random_engines[index]);
            pthread_mutex_unlock(&Q1_CTPs_mutex[current_state]);
        } else {
            pthread_mutex_lock(&Q2_CTPs_mutex[current_state]);
            new_state = Q2_CTPs[current_state](random_engines[index]);
            pthread_mutex_unlock(&Q2_CTPs_mutex[current_state]);
        }
        if (new_state >= current_state) {
            new_state += 1;
        }
        current_state = new_state;
    }
}

// Initialize simulation on each process
void init_simulation(const std::vector<std::string> & amino_acids, std::string pi_1_path, std::string Q_1_path, std::string pi_2_path, std::string Q_2_path) {
    amino_acids_alphabet = amino_acids;  
    for (std::string aa1 : amino_acids_alphabet) {
        for (std::string aa2 : amino_acids_alphabet) {
            amino_acids_pairs.push_back(aa1 + aa2);
        }
    }

    p1_probability_distribution = read_probability_distribution(pi_1_path, amino_acids_alphabet);
    p2_probability_distribution = read_probability_distribution(pi_2_path, amino_acids_pairs);
    Q1_rate_matrix = read_rate_matrix(Q_1_path, amino_acids_alphabet);
    Q2_rate_matrix = read_rate_matrix(Q_2_path, amino_acids_pairs);
    int num_states = amino_acids_alphabet.size();
    Q1_CTPs = new std::discrete_distribution<int>[num_states];
    Q2_CTPs = new std::discrete_distribution<int>[num_states * num_states];
    for(int current_state = 0; current_state < num_states; current_state++){
        std::vector<float> rate_vector = Q1_rate_matrix[current_state];
        rate_vector.erase(rate_vector.begin() + current_state);
        Q1_CTPs[current_state] = std::discrete_distribution(cbegin(rate_vector), cend(rate_vector));
    }
    for(int current_state = 0; current_state < num_states * num_states; current_state++){
        std::vector<float> rate_vector = Q2_rate_matrix[current_state];
        rate_vector.erase(rate_vector.begin() + current_state);
        Q2_CTPs[current_state] = std::discrete_distribution(cbegin(rate_vector), cend(rate_vector));
    }

    Q1_CTPs_mutex.clear();
    Q1_CTPs_mutex.resize(num_states);
    for(int current_state = 0; current_state < num_states; current_state++){
		pthread_mutex_init(&Q1_CTPs_mutex[current_state], NULL);
    }
    Q2_CTPs_mutex.clear();
    Q2_CTPs_mutex.resize(num_states * num_states);
    for(int current_state = 0; current_state < num_states * num_states; current_state++){
		pthread_mutex_init(&Q2_CTPs_mutex[current_state], NULL);
    }
}

// Run simulation for a family assigned to a certain process
void run_simulation(std::string tree_dir, std::string site_rates_dir, std::string contact_map_dir, std::string output_msa_dir, std::string family, int random_seed, int rank) {
    std::ofstream outfamproffile;
    std::string outfamproffilename = output_msa_dir + "/" + family + ".profiling";
    outfamproffile.open(outfamproffilename);
    outfamproffile << "Start run_simulation " << std::endl;
    outfamproffile << "Rank is " << rank << std::endl;
    outfamproffile << "The current family is " << family << std::endl;

    int numthreads = -1;
    #pragma omp parallel
    {
        numthreads = omp_get_num_threads();
    }
    outfamproffile << "The total number of threads is " << numthreads << std::endl;

    auto start_fam_sim = std::chrono::high_resolution_clock::now();

    std::string treefilepath = tree_dir + "/" + family + ".txt";
    std::string siteratefilepath = site_rates_dir + "/" + family + ".txt";
    std::string contactmapfilepath = contact_map_dir + "/" + family + ".txt";
    
    double read_tree_time_arr[1];
    double build_tree_time_arr[1];
    Tree currentTree = read_tree(treefilepath, read_tree_time_arr, build_tree_time_arr);
    double read_tree_time = read_tree_time_arr[0];
    double build_tree_time = build_tree_time_arr[0];

    auto start_reading_site_rates = std::chrono::high_resolution_clock::now();
    std::vector<float> site_rates_vec = read_site_rates(siteratefilepath);
    float site_rates[site_rates_vec.size()];
    copy(site_rates_vec.begin(), site_rates_vec.end(), site_rates);
    auto end_reading_site_rates = std::chrono::high_resolution_clock::now();

    auto start_reading_contact_map = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> contact_map = read_contact_map(contactmapfilepath);
    int num_sites = site_rates_vec.size();
    auto end_reading_contact_map = std::chrono::high_resolution_clock::now();

    auto end_reading = std::chrono::high_resolution_clock::now();
    
    // Further process sites
    std::vector<int> independent_sites_vec;
    std::set<int> contacting_sites;
    std::vector<int> contacting_sites_aux;
    std::vector<std::vector<int>> contacting_pairs;
    // Assume the contact map is symmetric
    for (int i = 0; i < num_sites; i++) {
        for (int j = i + 1; j < num_sites; j++) {
            if (contact_map[i][j] == 1) {
                std::vector<int> tmp;
                tmp.push_back(i);
                tmp.push_back(j);
                contacting_pairs.push_back(tmp);
                contacting_sites.insert(i);
                contacting_sites.insert(j);
                contacting_sites_aux.push_back(i);
                contacting_sites_aux.push_back(j);
            }
        }
    }
    // Check that each contacting site is in contact with exactly one other site.
    if(contacting_sites_aux.size() != contacting_sites.size()){
        std::cerr << "There are " << contacting_sites.size() << " unique contacting sites, but a total of " << contacting_sites_aux.size() << " contacting positions, so there must be repetitions!" << std::endl;
        exit(1);
    }
    for (int k = 0; k < num_sites; k++) {
        if (contacting_sites.find(k) == contacting_sites.end()) {
            independent_sites_vec.push_back(k);
        }
    }
    int independent_sites[independent_sites_vec.size()];
    copy(independent_sites_vec.begin(), independent_sites_vec.end(), independent_sites);
    int num_independent_sites = independent_sites_vec.size();
    int num_contacting_pairs = contacting_pairs.size();
    if(num_independent_sites + 2 * num_contacting_pairs != num_sites){
        std::cerr << "num_independent_sites and num_contacting_pairs dont add up: num_independent_sites = " << num_independent_sites << ", num_contacting_pairs = " << num_contacting_pairs << "; num_sites = " << num_sites << std::endl;
        exit(1);
    }

    // Generate random seeds, may generate a seed with current time if needed
    std::hash<std::string> stringHasher;
    size_t seed = stringHasher(family + std::to_string(random_seed));
    outfamproffile << "Seed for family " << family << " is " << seed << std::endl;
    std::srand(seed);
    int local_seed = std::rand();
    outfamproffile << "local_seed for family " << family << " is " << local_seed << std::endl;
    random_engines = new std::default_random_engine[num_independent_sites + num_contacting_pairs];
    // #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_independent_sites + num_contacting_pairs; i++) {
        std::default_random_engine generator_site(local_seed + i);
        random_engines[i] = generator_site;
    }

    auto end_processing_sites = std::chrono::high_resolution_clock::now();

    // Depth first search from root
    std::vector<std::string> dfs_order = currentTree.preorder_traversal();
    std::unordered_map<std::string, int> node_to_index_map;
    std::vector<std::vector<int>> msa_int(dfs_order.size(), std::vector<int>(num_independent_sites + num_contacting_pairs, 0));
    // Sample root state
    outfamproffile << "num_independent_sites " << num_independent_sites << std::endl;
    outfamproffile << "num_contacting_pairs " << num_contacting_pairs << std::endl;
    std::vector<int> root_states = sample_root_states(num_independent_sites, num_contacting_pairs);
    msa_int[0] = root_states;

    // Sample other nodes
    int parent_states_int[num_independent_sites + num_contacting_pairs];
    int node_states_int[num_independent_sites + num_contacting_pairs];
    std::string root = currentTree.root();
    for (int i = 0; i < int(dfs_order.size()); i++) {
        std::string node = dfs_order[i];
        node_to_index_map[node] = i;
        if (node == root) {
            continue;
        }
        adj_pair_t parent_pair = currentTree.parent(node);
        std::vector<int> & parent_states_int_vec = msa_int[node_to_index_map[parent_pair.node]];
        copy(parent_states_int_vec.begin(), parent_states_int_vec.end(), parent_states_int);
        float parent_pair_length = parent_pair.length;

        // Sample all the transitions for this node
        // First sample the independent sites
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < num_independent_sites; j++) {
            int starting_state = parent_states_int[j];
            float elapsed_time = parent_pair_length * site_rates[independent_sites[j]];
            node_states_int[j] = sample_transition(j, starting_state, elapsed_time, true);
        }
        // Then sample the contacting sites
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < num_contacting_pairs; j++) {
            int starting_state = parent_states_int[num_independent_sites + j];
            float elapsed_time = parent_pair_length;
            node_states_int[num_independent_sites + j] = sample_transition(num_independent_sites + j, starting_state, elapsed_time, false);
        }

        for(int j = 0; j < num_independent_sites + num_contacting_pairs; j++){
            msa_int[i][j] = node_states_int[j];
        }
    }

    auto end_sampling = std::chrono::high_resolution_clock::now();

    // Now translate the integer states back to amino acids
    std::map<std::string, std::vector<char>> msa;
    for (int k = 0; k < int(dfs_order.size()); k++) {
        std::string node = dfs_order[k];
        std::vector<int> & states_int = msa_int[k];
        std::vector<char> states(num_sites, ' ');
        // #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_independent_sites; i++) {
            int state_int = states_int[i];
            char state_str = amino_acids_alphabet[state_int].at(0);
            states[independent_sites[i]] = state_str;
        }
        // #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < num_contacting_pairs; j++) {
            int state_int = states_int[num_independent_sites + j];
            std::string state_str = amino_acids_pairs[state_int];
            states[contacting_pairs[j][0]] = state_str.at(0);
            states[contacting_pairs[j][1]] = state_str.at(1);
        }
        // #pragma omp parallel for schedule(dynamic)
        for (char s : states) {
            if (s == ' ') {
                std::cerr << "Error mapping integer states to amino acids." << std::endl;
                exit(1);
            }
        }
        msa[node] = states;
    }

    auto end_translating = std::chrono::high_resolution_clock::now();

    // Write back to files
    std::string msafilepath =  output_msa_dir + "/" + family + ".txt";
    write_msa(msafilepath, msa);

    auto end_fam_sim = std::chrono::high_resolution_clock::now();

    double reading_time = std::chrono::duration<double>(end_reading - start_fam_sim).count();
    outfamproffile << "Finish reading all the input files in " << reading_time << " seconds." << std::endl;

    outfamproffile << "Finish reading tree input files in " << read_tree_time << " seconds." << std::endl;
    outfamproffile << "Finish building tree in " << build_tree_time << " seconds." << std::endl;
    double reading_site_rates_time = std::chrono::duration<double>(end_reading_site_rates - start_reading_site_rates).count();
    outfamproffile << "Finish reading site_rates input files in " << reading_site_rates_time << " seconds." << std::endl;
    double reading_contact_map_time = std::chrono::duration<double>(end_reading_contact_map - start_reading_contact_map).count();
    outfamproffile << "Finish reading contact_map input files in " << reading_contact_map_time << " seconds." << std::endl;

    double processing_time = std::chrono::duration<double>(end_processing_sites - end_reading).count();
    outfamproffile << "Finish processing the data and other initialization in " << processing_time << " seconds." << std::endl;
    double sampling_time = std::chrono::duration<double>(end_sampling - end_processing_sites).count();
    outfamproffile << "Finish sampling in " << sampling_time << " seconds." << std::endl;
    double translating_time = std::chrono::duration<double>(end_translating - end_sampling).count();
    outfamproffile << "Finish translation in " << translating_time << " seconds." << std::endl;
    double writing_time = std::chrono::duration<double>(end_fam_sim - end_translating).count();
    outfamproffile << "Finish writing to the output file in " << writing_time << " seconds." << std::endl;
    double fam_time = std::chrono::duration<double>(end_fam_sim - start_fam_sim).count();
    outfamproffile << "Finish Simulation of " << family << " in " << fam_time << " seconds." << std::endl;
    outfamproffile.close();

    delete[] random_engines;
}


int main(int argc, char *argv[]) {
    // Init MPI
    auto start_all = std::chrono::high_resolution_clock::now();
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Start execution
        std::cout << "This is the start of this testing file ..." << std::endl;
        std::cout << "The number of process is " << num_procs << std::endl;
    }

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    // Read in all the arguments
    std::string tree_dir = argv[1];
    std::string site_rates_dir = argv[2];
    std::string contact_map_dir = argv[3];
    int num_of_families = std::atoi(argv[4]);
    int num_of_amino_acids = std::atoi(argv[5]);
    std::string pi_1_path = argv[6];
    std::string Q_1_path = argv[7];
    std::string pi_2_path = argv[8];
    std::string Q_2_path = argv[9];
    std::string strategy = argv[10];
    std::string output_msa_dir = argv[11];
    int random_seed = std::atoi(argv[12]);
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
    int load_balancing_mode = std::atoi(argv[13 + num_of_families + num_of_amino_acids]);
    if(load_balancing_mode > 0){
        std::string family_file_path = argv[13 + num_of_families + num_of_amino_acids + 1];
        families = read_family_sizes(families, family_file_path, load_balancing_mode, num_procs);
        assert(int(families.size()) == num_of_families);
    }

    std::ofstream outprofilingfile;
    std::ofstream outprofilingfile_local;


    // Initialize simulation
    init_simulation(amino_acids, pi_1_path, Q_1_path, pi_2_path, Q_2_path);

    MPI_Barrier(MPI_COMM_WORLD);

    auto end_init = std::chrono::high_resolution_clock::now();
    double init_time = std::chrono::duration<double>(end_init - start).count();
    if (rank == 0) {
        std::cout << "Finish Initializing in " << init_time << " seconds." << std::endl;
        std::string outputfilename =  output_msa_dir + "/" + "profiling.txt";
        outprofilingfile.open(outputfilename);
        outprofilingfile << "This is the start of this testing file ..." << std::endl;
        outprofilingfile << "The number of process is " << num_procs << std::endl;
        outprofilingfile << "Finish Initializing in " << init_time << " seconds." << std::endl;
    }

    std::string outputfilename_local =  output_msa_dir + "/profiling_" + std::to_string(rank) + ".txt";
    outprofilingfile_local.open(outputfilename_local);
    outprofilingfile_local << "This is the start of this testing file ..." << std::endl;
    outprofilingfile_local << "The number of process is " << num_procs << std::endl;
    outprofilingfile_local << "This is rank " << rank << std::endl;
    outprofilingfile_local << "Finish Initializing in " << init_time << " seconds." << std::endl;

    // Assign families to each rank.
    std::vector<std::string> local_families;
    for (int i = rank; i < num_of_families; i += num_procs) {
        local_families.push_back(families[i]);
    }

    // Run the simulation for all the families assigned to the process
    std::string msg;
    for (std::string family : local_families) {
        msg += " " + family;
    }
    if(DEBUG)
        std::cerr << "my families are: " << msg << std::endl;
    outprofilingfile_local << "my families are: " << msg << std::endl;
    int counter = 0;
    int tot = local_families.size();
    for (std::string family : local_families) {
        counter++;
        if(DEBUG)
            std::cerr << "Running on family " << rank << " " << family << " " << counter << "/" << tot << std::endl;
        run_simulation(tree_dir, site_rates_dir, contact_map_dir, output_msa_dir, family, random_seed, rank);
        if(DEBUG)
            std::cerr << "Done on family " << rank << " " << family << " " << counter << "/" << tot << std::endl;
    }

    delete[] Q1_CTPs;
    delete[] Q2_CTPs;

    auto end_sim_local = std::chrono::high_resolution_clock::now();
    double sim_time_local = std::chrono::duration<double>(end_sim_local - end_init).count();
    double entire_time_local = std::chrono::duration<double>(end_sim_local - start).count();
    outprofilingfile_local << "Finish Simulation in " << sim_time_local << " seconds." << std::endl;
    outprofilingfile_local << "Finish the entire program in " << entire_time_local << " seconds." << std::endl;
    outprofilingfile_local.close();

    if(DEBUG)
        std::cerr << "Finalizing " << rank << std::endl;
    MPI_Finalize();
    if(DEBUG)
        std::cerr << "Finalized!!! " << rank << std::endl;
    if (rank == 0) {
        auto end_all = std::chrono::high_resolution_clock::now();
        double entire_time = std::chrono::duration<double>(end_all - start_all).count();
        outprofilingfile << "Finish the entire program in " << entire_time << " seconds." << std::endl;
        outprofilingfile.close();
    }
}
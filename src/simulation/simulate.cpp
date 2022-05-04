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
 * argv[1]  (tree_dir): "/global/cscratch1/sd/sprillo/cs267_data/trees_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats"
 * argv[2]  (site_rates_dir): "/global/cscratch1/sd/sprillo/cs267_data/site_rates_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats"
 * argv[3]  (contact_map_dir): "/global/cscratch1/sd/sprillo/cs267_data/contact_maps_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats_maximal_matching"
 * argv[4]  (familiy_file_path): ./test_familiy_sizes.txt
 * argv[5]  (num_of_amino_acids): 20
 * argv[6]  (pi_1_path): "../../data/rate_matrices/wag_stationary.txt"
 * argv[7]  (Q_1_path): "../../data/rate_matrices/wag.txt"
 * agrv[8]  (pi_2_path): "../../data/rate_matrices/wag_x_wag_stationary.txt"
 * argv[9]  (Q_2_path): ".../../data/rate_matrices/wag_x_wag.txt"
 * argv[10] (strategy): "all_transitions"
 * argv[11] (output_msa_dir): "/global/cscratch1/sd/sprillo/xingyu_sim_out"
 * argv[12] (random_seed): 0
 * argv[13] (load_balancing_mode): 0 (0: naive version; 1: zig-zag)
 * argv[14 : 24] (amino_acids): A R N D C Q E G H I L K M F P S T W Y V
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

#include <omp.h>
#include <mpi.h>

// global variables
std::vector<std::string> families;
std::vector<float> p1_probability_distribution;
std::vector<float> p2_probability_distribution;
std::vector<std::vector<float>> Q1_rate_matrix;
std::vector<std::vector<float>> Q2_rate_matrix;
std::vector<std::string> amino_acids_alphabet;
std::vector<std::string> amino_acids_pairs;
std::default_random_engine* random_engines;

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
Tree read_tree(std::string treefilename) {
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
    }
    Tree newTree(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        treefile >> tmp;
        newTree.add_node(tmp);
    }
    treefile >> tmp;
    num_edges = std::stoi(tmp);
    treefile >> tmp;
    if (tmp != "edges") {
        std::cerr << "Tree file:" << treefilename << "should have line '[num_edges] edges' at position line " << num_nodes + 1 << std::endl;
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
        if (!newTree.is_node(u) || !newTree.is_node(v)) {
            std::cerr << "In Tree file " << treefilename << ": " << u << " and " << v << " should be nodes in the tree, but not." << std::endl;
        }
        newTree.add_edge(u, v, length);
    }
    if (num_edges != edges_count) {
        std::cerr << "Tree file:" << treefilename << "should have " << num_edges << " edges, but it has " << edges_count << " instead." << std::endl;
    }
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
    }
    while (siteratefile >> tmp) {
        result.push_back(std::stof(tmp));
    }
    if (result.size() != num_sites) {
        std::cerr << "Site rates file: " << filename << " was supposed to have " << num_sites << " sites, but it has " << result.size() << "." << std::endl;
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
    }
    
    while (contactmapfile >> tmp) {
        line_count += 1;
        std::vector<int> row(num_sites, 0);
        for (int i = 0; i < num_sites; i++) {
            char a = tmp[i];
            int c = std::atoi(&a);
            if (c != 0) {
                row[i] = 1;
            }
        }
        result.push_back(row);
    }
    if (num_sites != line_count) {
        std::cerr << "Contact map file: " << filename << " should have " << num_sites << " rows, but has " << line_count << "." << std::endl;
    }
    return result;
}

// Read the probability distribution
std::vector<float> read_probability_distribution(std::string filename, std::vector<std::string> element_list) {
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
    }
    tmpstring >> tmp2;
    if (tmp2 != "prob") {
        std::cerr << "Probability distribution file" << filename << "should have prob here but have " << tmp << " instead." << std::endl;
    }
    
    while(pfile.peek() != EOF) {
        std::string s;
        float p;

        getline(pfile, tmp);
        std::stringstream tmpstring(tmp);
        tmpstring >> s;
        states.push_back(s);
        tmpstring >> tmp2;
        p = std::stof(tmp2);
        sum += p;
        result.push_back(p);
    }
    
    float diff = std::abs(sum - 1.0);
    if (diff > 0.000001) {
        std::cout << "Probability distribution at " << filename << " should add to 1.0, with a tolerance of 1e-6." << std::endl;
    }

    if (states != element_list) {
        std::cerr << "Probability distribution file" << filename << " use a different (order of) alphabet." << std::endl;
    }
    return result;
}

// Read the rate matrix
std::vector<std::vector<float>> read_rate_matrix(std::string filename, std::vector<std::string> element_list) {
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
        std::stringstream tmpstring(tmp);
        tmpstring >> tmp2;
        states2.push_back(tmp2);
        while (tmpstring >> tmp2) {
            float p = std::stof(tmp2);
            row.push_back(p);
        }
        result.push_back(row);
    }

    if (states1 != element_list) {
        std::cerr << "Rate matrix file" << filename << " use a different (order of) alphabet." << std::endl;
    }

    if (states2 != element_list) {
        std::cerr << "Rate matrix file" << filename << " use a different (order of) alphabet." << std::endl;
    }

    return result;
}

// Read family sizes file
std::vector<std::string> read_family_sizes(std::string family_sizes_file, int load_balancing_mode = 1, int num_procs = 1) {
    std::vector<std::pair<int, std::string>> family_pairs;
    std::vector<std::string> result;
    std::string tmp, tmp1, tmp2, tmp3;

    std::ifstream famfile;
    famfile.open(family_sizes_file);

    getline(famfile, tmp);
    if (tmp != "family sequences sites") {
        std::cerr << "Family file" << family_sizes_file << " has a wrong format." << std::endl;
    }

    while (famfile.peek() != EOF) {
        getline(famfile, tmp);
        std::stringstream tmpstring(tmp);
        tmpstring >> tmp1;
        tmpstring >> tmp2;
        tmpstring >> tmp3;
        family_pairs.push_back(std::make_pair(std::stoi(tmp2) * std::stoi(tmp3), tmp1));
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
        for (int i = 2 * num_procs * std::floor(family_pairs.size() / (2 * num_procs)); i < family_pairs.size(); i += 1) {
            result.push_back(family_pairs[i].second);
        }
    }
    
    return result;
}

// Write msa files
void write_msa(std::string filename, std::map<std::string, std::vector<std::string>> msa) {
    std::ofstream outfile;
    outfile.open(filename);

    for (auto key_value : msa) {
        outfile << ">" << key_value.first << std::endl;
        std::string tmp = "";
        for (std::string s : key_value.second) {
            tmp = tmp + s;
        }
        outfile <<tmp << std::endl;
    }

    outfile.close();
}

// Sample root state
std::vector<int> sample_root_states(int num_independent_sites, int num_contacting_pairs) {
    std::vector<int> result(num_independent_sites + num_contacting_pairs, 0);

    #pragma omp parallel
    {
        int threadnum = omp_get_thread_num();
        int numthreads = omp_get_num_threads();
        // First sample the independent sites
        std::discrete_distribution distribution1(cbegin(p1_probability_distribution), cend(p1_probability_distribution));
        for (int i = threadnum; i < num_independent_sites; i += numthreads) {
            result[i] = distribution1(random_engines[i]);
        }

        // Then sample the contacting sites
        std::discrete_distribution distribution2(cbegin(p2_probability_distribution), cend(p2_probability_distribution));
        for (int j = threadnum; j < num_contacting_pairs; j += numthreads) {
            result[num_independent_sites + j] = distribution2(random_engines[num_independent_sites + j]);
        }
    }

    return result;
}

// Sample a transition
int sample_transition(int index, int starting_state, float elapsed_time, std::string strategy, bool if_independent) {
    if (strategy != "all_transitions") {
        std::cerr << "Unknown strategy: " << strategy << std::endl;
        return -1;
    }
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
        std::vector<float> rate_vector;
        if (if_independent) {
            rate_vector = Q1_rate_matrix[current_state];
        } else {
            rate_vector = Q2_rate_matrix[current_state];
        }
        rate_vector.erase(rate_vector.begin() + current_state);
        std::discrete_distribution distribution2(cbegin(rate_vector), cend(rate_vector));
        int new_state = distribution2(random_engines[index]);
        if (new_state >= current_state) {
            new_state += 1;
        }
        current_state = new_state;
    }
}

// Initialize simulation on each process
void init_simulation(std::vector<std::string> amino_acids, std::string family_sizes_file, int load_balancing_mode, int num_procs, std::string pi_1_path, std::string Q_1_path, std::string pi_2_path, std::string Q_2_path) {
    amino_acids_alphabet = amino_acids;  
    for (std::string aa1 : amino_acids_alphabet) {
        for (std::string aa2 : amino_acids_alphabet) {
            amino_acids_pairs.push_back(aa1 + aa2);
        }
    }

    families = read_family_sizes(family_sizes_file, load_balancing_mode, num_procs);
    p1_probability_distribution = read_probability_distribution(pi_1_path, amino_acids_alphabet);
    p2_probability_distribution = read_probability_distribution(pi_2_path, amino_acids_pairs);
    Q1_rate_matrix = read_rate_matrix(Q_1_path, amino_acids_alphabet);
    Q2_rate_matrix = read_rate_matrix(Q_2_path, amino_acids_pairs);
}

// Run simulation for a family assigned to a certain process
void run_simulation(std::string tree_dir, std::string site_rates_dir, std::string contact_map_dir, std::string output_msa_dir, std::string family, int random_seed, std::string strategy) {
    std::ofstream outfamproffile;
    std::string outfamproffilename = output_msa_dir + "/" + family + ".profiling";
    outfamproffile.open(outfamproffilename);
    outfamproffile << "The current family is " << family << std::endl;
    
    int numthreads;
    #pragma omp parallel
    {
        numthreads = omp_get_num_threads();
    }
    outfamproffile << "The total number of threads is " << numthreads << std::endl;

    auto start_fam_sim = std::chrono::high_resolution_clock::now();

    std::string treefilepath = tree_dir + "/" + family + ".txt";
    std::string siteratefilepath = site_rates_dir + "/" + family + ".txt";
    std::string contactmapfilepath = contact_map_dir + "/" + family + ".txt";
    
    Tree currentTree = read_tree(treefilepath);
    std::vector<float> site_rates = read_site_rates(siteratefilepath);
    std::vector<std::vector<int>> contact_map = read_contact_map(contactmapfilepath);
    int num_sites = site_rates.size();

    auto end_reading = std::chrono::high_resolution_clock::now();
    
    // Further process sites
    std::vector<int> independent_sites;
    std::set<int> contacting_sites;
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
            }
        }
    }
    for (int k = 0; k < num_sites; k++) {
        if (contacting_sites.find(k) == contacting_sites.end()) {
            independent_sites.push_back(k);
        }
    }
    int num_independent_sites = independent_sites.size();
    int num_contacting_pairs = contacting_pairs.size();

    // Generate random seeds, may generate a seed with current time if needed
    std::hash<std::string> stringHasher;
    size_t seed = stringHasher(family + std::to_string(random_seed));
    std::srand(seed);
    int local_seed = std::rand();
    random_engines = new std::default_random_engine[num_independent_sites + num_contacting_pairs];
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_independent_sites + num_contacting_pairs; i++) {
        std::default_random_engine generator_site(local_seed + i);
        random_engines[i] = generator_site;
    }

    auto end_processing_sites = std::chrono::high_resolution_clock::now();

    // Depth first search from root
    std::vector<std::string> dfs_order = currentTree.preorder_traversal();
    std::unordered_map<std::string, int> node_to_index_map;
    std::vector<std::vector<int>> msa_int;
    // Sample root state
    std::vector<int> root_states = sample_root_states(num_independent_sites, num_contacting_pairs);
    msa_int.push_back(root_states);

    // Sample other nodes
    for (int i = 0; i < dfs_order.size(); i++) {
        std::string node = dfs_order[i];
        node_to_index_map[node] = i;
        if (node == currentTree.root()) {
            continue;
        }
        std::vector<int> node_states_int(num_independent_sites + num_contacting_pairs, 0);
        adj_pair_t parent_pair = currentTree.parent(node);
        std::vector<int> parent_states_int = msa_int[node_to_index_map[parent_pair.node]];

        // Sample all the transitions for this node
        // First sample the independent sites
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_independent_sites; i++) {
            int starting_state = parent_states_int[i];
            float elapsed_time = parent_pair.length * site_rates[independent_sites[i]];
            node_states_int[i] = sample_transition(i, starting_state, elapsed_time, strategy, true);
        }
        // Then sample the contacting sites
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < num_contacting_pairs; j++) {
            int starting_state = parent_states_int[num_independent_sites + j];
            float elapsed_time = parent_pair.length;
            node_states_int[num_independent_sites + j] = sample_transition(num_independent_sites + j, starting_state, elapsed_time, strategy, false);
        }

        msa_int.push_back(node_states_int);
    }

    auto end_sampling = std::chrono::high_resolution_clock::now();

    // Now translate the integer states back to amino acids
    std::map<std::string, std::vector<std::string>> msa;
    for (int k = 0; k < dfs_order.size(); k++) {
        std::string node = dfs_order[k];
        std::vector<int> states_int = msa_int[k];
        std::vector<std::string> states(num_sites, "");
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_independent_sites; i++) {
            int state_int = states_int[i];
            std::string state_str = amino_acids_alphabet[state_int];
            states[independent_sites[i]] = state_str;
        }
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < num_contacting_pairs; j++) {
            int state_int = states_int[num_independent_sites + j];
            std::string state_str = amino_acids_pairs[state_int];
            states[contacting_pairs[j][0]] = state_str[0];
            states[contacting_pairs[j][1]] = state_str[1];
        }
        #pragma omp parallel for schedule(dynamic)
        for (std::string s : states) {
            if (s == "") {
                std::cerr << "Error mapping integer states to amino acids." << std::endl;
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
}


int main(int argc, char *argv[]) {
    // Init MPI
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
    std::string family_file_path = argv[4];
    int num_of_amino_acids = std::atoi(argv[5]);
    std::string pi_1_path = argv[6];
    std::string Q_1_path = argv[7];
    std::string pi_2_path = argv[8];
    std::string Q_2_path = argv[9];
    std::string strategy = argv[10];
    std::string output_msa_dir = argv[11];
    int random_seed = std::atoi(argv[12]);
    int load_balancing_mode = std::atoi(argv[13]);
    std::vector<std::string> amino_acids;
    amino_acids.reserve(num_of_amino_acids);
    for (int i = 0; i < num_of_amino_acids; i++) {
        amino_acids.push_back(argv[14 + i]);
    }

    std::ofstream outprofilingfile;


    // Initialize simulation
    init_simulation(amino_acids, family_file_path, load_balancing_mode, num_procs, pi_1_path, Q_1_path, pi_2_path, Q_2_path);

    auto end_init = std::chrono::high_resolution_clock::now();
    double init_time = std::chrono::duration<double>(end_init - start).count();
    if (rank == 0) {
        std::cout << " Rank 0 finish Initializing in " << init_time << " seconds." << std::endl;
    }

    std::string outputfilename =  output_msa_dir + "/profiling_" + std::to_string(rank) + ".txt";
    outprofilingfile.open(outputfilename);
    outprofilingfile << "This is the start of this testing file ..." << std::endl;
    outprofilingfile << "The number of process is " << num_procs << std::endl;
    outprofilingfile << "This is rank " << rank << std::endl;
    outprofilingfile << "Finish Initializing in " << init_time << " seconds." << std::endl;




    // Assign families to each rank
    // Currently, we just statically "evenly" assign family to each rank before simulation
    // There might be some dynamic load balancing techniques here.
    std::vector<std::string> local_families;
    for (int i = rank; i < families.size(); i += num_procs) {
        local_families.push_back(families[i]);
    }


    // Run the simulation for all the families assigned to the process
    for (std::string family : local_families) {
        run_simulation(tree_dir, site_rates_dir, contact_map_dir, output_msa_dir, family, random_seed + rank, strategy);
    }

    auto end_sim = std::chrono::high_resolution_clock::now();
    double sim_time = std::chrono::duration<double>(end_sim - end_init).count();
    double entire_time = std::chrono::duration<double>(end_sim - start).count();
    if (rank == 0) {
        std::cout << " Rank 0 finish Simulation in " << sim_time << " seconds." << std::endl;
        std::cout << " Rank 0 finish the entire program in " << entire_time << " seconds." << std::endl;
    }

    outprofilingfile << "Finish Simulation in " << sim_time << " seconds." << std::endl;
    outprofilingfile << "Finish the entire program in " << entire_time << " seconds." << std::endl;
    outprofilingfile.close();

    MPI_Finalize();
}
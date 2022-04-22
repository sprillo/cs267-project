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
#include <sstream>
#include <random>
#include <vector>
#include <unordered_map>
#include <set>
#include <mpi.h>

// global variables
std::vector<float> p1_probability_distribution;
std::vector<float> p2_probability_distribution;
std::vector<std::vector<float>> Q1_rate_matrix;
std::vector<std::vector<float>> Q2_rate_matrix;
std::vector<std::string> amino_acids_alphabet;
std::vector<std::string> amino_acids_pairs;

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

    std::fstream treefile;
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

    std::fstream siteratefile;
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

    std::fstream contactmapfile;
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
        // if (row.size() != 1000) {
        //     std::cerr << "Error " << row.size() << " at line " << std::endl;
        // }
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

    std::fstream pfile;
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

    std::fstream qfile;
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

    // for (auto state : states1) {
    //     std::cout << state << " ";
    // }
    // std::cout << " " << std::endl;
    // for (auto state : states2) {
    //     std::cout << state << std::endl;
    // }
    // for (auto p : result) {
    //     for (auto q : p) {
    //         std::cout << q << " ";
    //     }
    //     std::cout << "" << std::endl;
    // }

    return result;
}



// Initialize simulation on each process
void init_simulation(std::vector<std::string> amino_acids, std::string pi_1_path, std::string Q_1_path, std::string pi_2_path, std::string Q_2_path) {
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
}

// Run simulation for a family assigned to a certain process
void run_simulation(std::string tree_dir, std::string site_rates_dir, std::string contact_map_dir, std::string family) {
    std::cout << "The current family is " << family << std::endl;
    std::string treefilepath = tree_dir + "/" + family + ".txt";
    std::string siteratefilepath = site_rates_dir + "/" + family + ".txt";
    std::string contactmapfilepath = contact_map_dir + "/" + family + ".txt";
    
    Tree currentTree = read_tree(treefilepath);
    std::vector<float> site_rates = read_site_rates(siteratefilepath);
    std::vector<std::vector<int>> contact_map = read_contact_map(contactmapfilepath);
    int num_sites = site_rates.size();
    
    // Further process sites
    std::set<int> independent_sites;
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
            independent_sites.insert(k);
        }
    }
    int num_independent_sites = independent_sites.size();
    int num_contacting_sites = contacting_sites.size();
    
    

    
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
    init_simulation(amino_acids, pi_1_path, Q_1_path, pi_2_path, Q_2_path);

    // Run the simulation for all the families assigned to the process
    for (std::string family : families) {
        run_simulation(tree_dir, site_rates_dir, contact_map_dir, family);
    }
    


    // MPI_Finalize();
}
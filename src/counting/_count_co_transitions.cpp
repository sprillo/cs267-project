#include <vector>
#include<string>
#include <sstream> 
#include <algorithm>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <map>
#include <utility> 
#include <chrono>
#include <omp.h>

#define PROFILE true

using namespace std;

// prifiling
double total_time = 0;
double time_parse_param = 0;
double time_init_aa_pairs = 0;
double time_init_count_matrices_data = 0;
// double time_read_tree = 0;
// double time_read_msa = 0;
// double time_read_contact_map = 0;
// double time_compute_contacting_pairs = 0;
// double time_compute_count_matrices = 0;
double time_read_dataset_and_compute_count_matrices;
double time_create_count_matrices_datastructure = 0;
double time_write_count_matrices = 0;
auto start_ = std::chrono::high_resolution_clock::now();
auto end_ = std::chrono::high_resolution_clock::now();
auto start_time = std::chrono::high_resolution_clock::now();
auto end_time = std::chrono::high_resolution_clock::now();

vector<string> pairs_of_amino_acids;
map<string, int> aa_pair_to_int;;
int num_of_amino_acids;
map<string, string> msa;
int num_sites;
int count_matrix_size;
int num_quantization_points;
int num_of_families;

struct count_matrix{
    double q;
    double* matrix;
};

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
Tree* read_tree(std::string treefilename) {
    int num_nodes;
    int num_edges;
    int edges_count = 0;
    std::string tmp;

    std::fstream treefile;
    treefile.open(treefilename, ios::in);
    treefile >> tmp;
    num_nodes = std::stoi(tmp);
    treefile >> tmp;
    if (tmp != "nodes") {
        std::cerr << "Tree file:" << treefilename << "should start with '[num_nodes] nodes'." << std::endl;
    }
    Tree* newTree = new Tree(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        treefile >> tmp;
        newTree->add_node(tmp);
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
        if (!newTree->is_node(u) || !newTree->is_node(v)) {
            std::cerr << "In Tree file " << treefilename << ": " << u << " and " << v << " should be nodes in the tree, but not." << std::endl;
        }
        newTree->add_edge(u, v, length);
    }
    if (num_edges != edges_count) {
        std::cerr << "Tree file:" << treefilename << "should have " << num_edges << " edges, but it has " << edges_count << " instead." << std::endl;
    }
    return newTree;
}

map<string, string>* read_msa(const string & filename){
    map<string, string>* msa = new map<string, string>;
    std::string tmp;
    std::fstream file;
    file.open(filename);
    while (file.peek() != EOF) {
        getline(file, tmp);
        string name = tmp.substr(1);
        getline(file, tmp);
        (*msa)[name] = tmp;
    }
    file.close();
    return msa;
}

char* read_contact_map(const string & filename){
    std::string tmp;
    std::fstream file;
    file.open(filename, ios::in);
    getline(file, tmp);
    num_sites = stoi(tmp.substr(0, tmp.find(' ')));
    char* buffer = (char *)malloc(num_sites * (num_sites+1) * sizeof(char));
    file.read(buffer, num_sites * (num_sites+1));
    file.close();
    return buffer;
}

int quantization_idx(float branch_length, const vector<float> & quantization_points_sorted){
    if (branch_length < quantization_points_sorted[0] || branch_length > quantization_points_sorted.back()) return -1;
    int smallest_upper_bound_idx = lower_bound(quantization_points_sorted.begin(), quantization_points_sorted.end(), branch_length) - quantization_points_sorted.begin();
    if (smallest_upper_bound_idx == 0) return 0;
    else{
        float left_value = quantization_points_sorted[smallest_upper_bound_idx - 1];
        float right_value = quantization_points_sorted[smallest_upper_bound_idx];
        float relative_error_left = branch_length / left_value - 1;
        float relative_error_right = right_value / branch_length - 1;
        if (relative_error_left < relative_error_right) return smallest_upper_bound_idx - 1;
        else return smallest_upper_bound_idx;
    }
}

bool all_children_are_leafs(Tree* tree, const vector<adj_pair_t> & children){
    for (const adj_pair_t& p : children){
        if (!tree->is_leaf(p.node)) return false;
    }
    return true;
}

bool my_find(const vector<string> & amino_acids, const string & key){
    for (int i=0; i<amino_acids.size(); i++){
        if (amino_acids[i] == key) return true;
    }
    return false;
}

vector<count_matrix> _map_func(
    const string & tree_dir,
    const string & msa_dir,
    const string & contact_map_dir,
    const string* families,
    const vector<string> & amino_acids,
    const vector<float> & quantization_points,
    const string & edge_or_cherry,
    int minimum_distance_for_nontrivial_contact
){
    if (PROFILE) start_ = std::chrono::high_resolution_clock::now();
    vector<count_matrix> count_matrices;
    num_quantization_points = quantization_points.size();
    count_matrix_size = pairs_of_amino_acids.size();
    int count_matrix_num_entries = count_matrix_size * count_matrix_size;
    double* count_matrices_data = new double[num_quantization_points * count_matrix_num_entries];
    for (int i=0; i<num_quantization_points * count_matrix_num_entries; i++){
        count_matrices_data[i] = 0;
    }
    if (PROFILE) end_ = std::chrono::high_resolution_clock::now();
    if (PROFILE) time_init_count_matrices_data += std::chrono::duration<double>(end_ - start_).count();

    if (PROFILE) start_ = std::chrono::high_resolution_clock::now();

    for (int i=0; i<num_of_families; i++){
        const string & family = families[i];
        // if (PROFILE) start_ = std::chrono::high_resolution_clock::now();
        Tree* tree = read_tree(tree_dir + "/" + family + ".txt");
        // if (PROFILE) end_ = std::chrono::high_resolution_clock::now();
        // if (PROFILE) time_read_tree += std::chrono::duration<double>(end_ - start_).count();
        
        // if (PROFILE) start_ = std::chrono::high_resolution_clock::now();
        map<string, string>* msa = read_msa(msa_dir + "/" + family + ".txt");
        // if (PROFILE) end_ = std::chrono::high_resolution_clock::now();
        // if (PROFILE) time_read_msa += std::chrono::duration<double>(end_ - start_).count();

        // if (PROFILE) start_ = std::chrono::high_resolution_clock::now();
        char* contact_map = read_contact_map(contact_map_dir + "/" + family + ".txt");
        // if (PROFILE) end_ = std::chrono::high_resolution_clock::now();
        // if (PROFILE) time_read_contact_map += std::chrono::duration<double>(end_ - start_).count();

        // if (PROFILE) start_ = std::chrono::high_resolution_clock::now();
        // vector<pair<int, int>> contacting_pairs;
        // contacting_pairs.reserve(num_sites * num_sites / 2);
        pair<int, int>* contacting_pairs = new pair<int, int>[num_sites * num_sites / 2];
        int num_contacting_pairs = 0;
        for (int i=0; i<num_sites; i++){
            for (int j=i+minimum_distance_for_nontrivial_contact; j<num_sites; j++){
                if (contact_map[i*(num_sites+1)+j] == '1'){
                    contacting_pairs[num_contacting_pairs] = pair<int, int>(i, j);
                    num_contacting_pairs++;
                }
            }
        }
        // pair<int, int>* contacting_pairs = new pair<int, int>[num_contacting_pairs];
        // for (int i=0; i<num_sites; i++){
        //     for (int j=i+minimum_distance_for_nontrivial_contact; j<num_sites; j++){
        //         if (contact_map[i*(num_sites+1)+j] == '1'){
                    
        //         }
        //     }
        // } 
        // if (PROFILE) end_ = std::chrono::high_resolution_clock::now();
        // if (PROFILE) time_compute_contacting_pairs += std::chrono::duration<double>(end_ - start_).count();

        // if (PROFILE) start_ = std::chrono::high_resolution_clock::now();
        // #pragma omp parallel for
        #pragma omp parallel for
        for (string node : tree->nodes()){
            if (edge_or_cherry == "edge") {
                string node_seq = (*msa)[node];
                for (adj_pair_t& edge : tree->children(node)){
                    string child = edge.node;
                    float branch_length = edge.length;
                    string child_seq = (*msa)[child];
                    int q_idx = quantization_idx(branch_length, quantization_points);
                    if (q_idx != -1){
                        // #pragma omp parallel for
                        for (int k=0; k<num_contacting_pairs; k++){
                            pair<int, int>& p = contacting_pairs[k];
                            int i = p.first;
                            int j = p.second;
                            string start_state = string{node_seq[i], node_seq[j]};
                            string end_state = string{child_seq[i], child_seq[j]};
                            if (
                                // my_find(amino_acids, string{node_seq[i]})
                                // && my_find(amino_acids, string{node_seq[j]})
                                // && my_find(amino_acids, string{child_seq[i]})
                                // && my_find(amino_acids, string{child_seq[j]})
                                find(amino_acids.begin(), amino_acids.end(), string{node_seq[i]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{node_seq[j]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{child_seq[i]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{child_seq[j]}) != amino_acids.end()
                            ){
                                #pragma omp atomic
                                count_matrices_data[q_idx * count_matrix_num_entries + aa_pair_to_int[start_state] * count_matrix_size + aa_pair_to_int[end_state]] += 0.5;
                                reverse(start_state.begin(), start_state.end());
                                reverse(end_state.begin(), end_state.end());
                                #pragma omp atomic
                                count_matrices_data[q_idx * count_matrix_num_entries + aa_pair_to_int[start_state] * count_matrix_size + aa_pair_to_int[end_state]] += 0.5;
                            }
                        }
                    }
                }
            } else { // cherry
                vector<adj_pair_t> children = tree->children(node);
                if (children.size() == 2 && all_children_are_leafs(tree, children)){
                    string leaf_1 = children[0].node;
                    float branch_length_1 = children[0].length;
                    string leaf_2 = children[1].node;
                    float branch_length_2 = children[1].length;
                    string leaf_seq_1 = (*msa)[leaf_1];
                    string leaf_seq_2 = (*msa)[leaf_2];
                    float branch_length_total = branch_length_1 + branch_length_2;
                    int q_idx = quantization_idx(branch_length_total, quantization_points);
                    if (q_idx != -1){
                        // #pragma omp parallel for
                        for (int k=0; k<num_contacting_pairs; k++){
                            pair<int, int>& p = contacting_pairs[k];
                            int i = p.first;
                            int j = p.second;
                            string start_state = string{leaf_seq_1[i], leaf_seq_1[j]};
                            string end_state = string{leaf_seq_2[i], leaf_seq_2[j]};
                            if (
                                // my_find(amino_acids, string{leaf_seq_1[i]})
                                // && my_find(amino_acids, string{leaf_seq_1[j]})
                                // && my_find(amino_acids, string{leaf_seq_2[i]})
                                // && my_find(amino_acids, string{leaf_seq_2[j]})
                                find(amino_acids.begin(), amino_acids.end(), string{leaf_seq_1[i]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{leaf_seq_1[j]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{leaf_seq_2[i]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{leaf_seq_2[j]}) != amino_acids.end()
                            ){
                                #pragma omp atomic
                                count_matrices_data[q_idx * count_matrix_num_entries + aa_pair_to_int[start_state] * count_matrix_size + aa_pair_to_int[end_state]] += 0.25;
                                #pragma omp atomic
                                count_matrices_data[q_idx * count_matrix_num_entries + aa_pair_to_int[end_state] * count_matrix_size + aa_pair_to_int[start_state]] += 0.25;
                                reverse(start_state.begin(), start_state.end());
                                reverse(end_state.begin(), end_state.end());
                                #pragma omp atomic
                                count_matrices_data[q_idx * count_matrix_num_entries + aa_pair_to_int[start_state] * count_matrix_size + aa_pair_to_int[end_state]] += 0.25;
                                #pragma omp atomic
                                count_matrices_data[q_idx * count_matrix_num_entries + aa_pair_to_int[end_state] * count_matrix_size + aa_pair_to_int[start_state]] += 0.25;
                            }
                        }
                    }
                }
            }
        }
        // if (PROFILE) end_ = std::chrono::high_resolution_clock::now();
        // if (PROFILE) time_compute_count_matrices += std::chrono::duration<double>(end_ - start_).count();
    }
    if (PROFILE) end_ = std::chrono::high_resolution_clock::now();
    if (PROFILE) time_read_dataset_and_compute_count_matrices += std::chrono::duration<double>(end_ - start_).count();

    if (PROFILE) start_ = std::chrono::high_resolution_clock::now();
    for (int i=0; i<quantization_points.size(); i++){
        count_matrices.push_back(count_matrix{quantization_points[i], &(count_matrices_data[i * count_matrix_num_entries])});
    }
    if (PROFILE) end_ = std::chrono::high_resolution_clock::now();
    if (PROFILE) time_create_count_matrices_datastructure += std::chrono::duration<double>(end_ - start_).count();

    return count_matrices;
}

void write_count_matrices(const vector<count_matrix> & count_matrices, const string & output_count_matrices_dir){
    std::ofstream myfile;
    myfile.open (output_count_matrices_dir);
    myfile << num_quantization_points << " matrices\n" << count_matrix_size << " states\n";
    for (const count_matrix& cm : count_matrices){
        myfile << cm.q << "\n";
        myfile << "\t";
        for (string& pair : pairs_of_amino_acids){
            myfile << pair << "\t";
        }
        myfile << "\n";
        for (int i=0; i<count_matrix_size; i++){
            myfile << pairs_of_amino_acids[i] << "\t";
            for (int j=0; j<count_matrix_size; j++){
                myfile << cm.matrix[i * count_matrix_size + j];
                if (j != count_matrix_size-1){
                    myfile << "\t";
                }
            }
            myfile << "\n";
        }
    }
    myfile.close();
}

void count_co_transitions(
    const string & tree_dir,
    const string & msa_dir,
    const string & contact_map_dir,
    const string * families,
    const vector<string> & amino_acids,
    vector<float>& quantization_points,
    const string & edge_or_cherry,
    int minimum_distance_for_nontrivial_contact,
    const string & output_count_matrices_dir,
    int num_processes){

    if (PROFILE) start_ = std::chrono::high_resolution_clock::now();

    sort(quantization_points.begin(), quantization_points.end());
    for (const string& a1 : amino_acids){
        for (const string& a2 : amino_acids){
            pairs_of_amino_acids.push_back(a1+a2);
        }
    }
    for (int i=0; i<pairs_of_amino_acids.size(); i++){
        aa_pair_to_int[pairs_of_amino_acids[i]] = i;
    }
    
    if (PROFILE) end_ = std::chrono::high_resolution_clock::now();
    if (PROFILE) time_init_aa_pairs += std::chrono::duration<double>(end_ - start_).count();

    vector<count_matrix> count_matrices = _map_func(tree_dir, msa_dir, contact_map_dir, families, 
            amino_acids, quantization_points, edge_or_cherry, minimum_distance_for_nontrivial_contact);

    if (PROFILE) start_ = std::chrono::high_resolution_clock::now();
    write_count_matrices(count_matrices, output_count_matrices_dir + "/result.txt");
    if (PROFILE) end_ = std::chrono::high_resolution_clock::now();
    if (PROFILE) time_write_count_matrices += std::chrono::duration<double>(end_ - start_).count();
}

int main(int argc, char *argv[]) {
    // Read in all the arguments
    if (PROFILE) start_time = std::chrono::high_resolution_clock::now();
    if (PROFILE) start_ = std::chrono::high_resolution_clock::now();

    string tree_dir = argv[1];
    string msa_dir = argv[2];
    string contact_map_dir = argv[3];
    num_of_families = atoi(argv[4]);
    num_of_amino_acids = atoi(argv[5]);
    int num_of_quantization_points = atoi(argv[6]);
    string* families = new string[num_of_families];
    for (int i = 0; i < num_of_families; i++) {
        families[i] = argv[7 + i];
    }
    vector<string> amino_acids;
    amino_acids.reserve(num_of_amino_acids);
    for (int i = 0; i < num_of_amino_acids; i++) {
        amino_acids.push_back(argv[7 + num_of_families + i]);
    }
    vector<float> quantization_points;
    quantization_points.reserve(num_of_quantization_points);
    for (int i = 0; i < num_of_quantization_points; i++) {
        quantization_points.push_back(atof(argv[7 + num_of_families + num_of_amino_acids + i]));
    } 
    string edge_or_cherry = argv[7 + num_of_families + num_of_amino_acids + num_of_quantization_points];
    int minimum_distance_for_nontrivial_contact = atoi(argv[7 + num_of_families + num_of_amino_acids + num_of_quantization_points + 1]);
    string output_count_matrices_dir = argv[7 + num_of_families + num_of_amino_acids + num_of_quantization_points + 2];
    int num_processes = atoi(argv[7 + num_of_families + num_of_amino_acids + num_of_quantization_points + 3]);
    
    if (PROFILE) end_ = std::chrono::high_resolution_clock::now();
    if (PROFILE) time_parse_param += std::chrono::duration<double>(end_ - start_).count();
    count_co_transitions(
        tree_dir,
        msa_dir,
        contact_map_dir,
        families,
        amino_acids,
        quantization_points,
        edge_or_cherry,
        minimum_distance_for_nontrivial_contact,
        output_count_matrices_dir,
        num_processes
    );

    if (PROFILE) end_time = std::chrono::high_resolution_clock::now();
    if (PROFILE) total_time += std::chrono::duration<double>(end_time - start_time).count();
    if (PROFILE) cout << "Proliling:" << endl;
    if (PROFILE) cout << "time_parse_param: " << time_parse_param << endl;
    if (PROFILE) cout << "time_init_aa_pairs: " << time_init_aa_pairs << endl;
    if (PROFILE) cout << "time_init_count_matrices_data: " << time_init_count_matrices_data << endl;
    // if (PROFILE) cout << "time_read_tree: " << time_read_tree << endl;
    // if (PROFILE) cout << "time_read_msa: " << time_read_msa << endl;
    // if (PROFILE) cout << "time_read_contact_map: " << time_read_contact_map << endl;
    // if (PROFILE) cout << "time_compute_contacting_pairs: " << time_compute_contacting_pairs << endl;
    // if (PROFILE) cout << "time_compute_count_matrices: " << time_compute_count_matrices << endl;
    if (PROFILE) cout << "time_read_dataset_and_compute_count_matrices: " << time_read_dataset_and_compute_count_matrices << endl;
    if (PROFILE) cout << "time_create_count_matrices_datastructure: " << time_create_count_matrices_datastructure << endl;
    if (PROFILE) cout << "time_write_count_matrices: " << time_write_count_matrices << endl;
    if (PROFILE) cout << "Total time: " << total_time << endl;
}
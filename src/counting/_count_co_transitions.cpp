#include <vector>
#include<string>
#include <sstream> 
#include <algorithm>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <map>
#include <utility> 

using namespace std;

vector<string> pairs_of_amino_acids;
map<string, int> aa_pair_to_int;;
int num_of_amino_acids;
map<string, string> msa;

struct count_matrix{
    double q;
    vector<vector<double>> matrix;
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

map<string, string> read_msa(const string & filename){
    map<string, string> msa;
    std::string tmp;
    std::fstream file;
    file.open(filename);
    while (file.peek() != EOF) {
        getline(file, tmp);
        string name = tmp.substr(1);
        getline(file, tmp);
        msa[name] = tmp;
    }
    file.close();
    return msa;
}

vector<vector<bool>> read_contact_map(const string & filename){
    vector<vector<bool>> contact_map;
    std::string tmp;
    std::fstream file;
    file.open(filename);
    getline(file, tmp);
    int num_sites = stoi(tmp.substr(0, tmp.find(' ')));
    contact_map.resize(num_sites);
    for (int i=0; i<num_sites; i++){ 
        contact_map[i].resize(num_sites);
        getline(file, tmp);
        for (int j=0; j<num_sites; j++){
            contact_map[i][j] = tmp[j] == '1';
        }
    }
    file.close();
    return contact_map;
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

bool all_children_are_leafs(Tree & tree, const vector<adj_pair_t> & children){
    for (const adj_pair_t& p : children){
        if (!tree.is_leaf(p.node)) return false;
    }
    return true;
}


vector<count_matrix> _map_func(
    const string & tree_dir,
    const string & msa_dir,
    const string & contact_map_dir,
    const vector<string> & families,
    const vector<string> & amino_acids,
    const vector<float> & quantization_points,
    const string & edge_or_cherry,
    int minimum_distance_for_nontrivial_contact
){
    vector<count_matrix> count_matrices;
    vector<vector<vector<double>>> count_matrices_data;
    count_matrices_data.resize(quantization_points.size());
    for (auto & matrix : count_matrices_data){
        matrix.resize(pairs_of_amino_acids.size());
        for (auto & row : matrix){
            row.resize(pairs_of_amino_acids.size());
        }
    }

    for (const string & family : families){
        Tree tree = read_tree(tree_dir + "/" + family + ".txt");
        map<string, string> msa = read_msa(msa_dir + "/" + family + ".txt");
        vector<vector<bool>> contact_map = read_contact_map(contact_map_dir + "/" + family + ".txt");
        vector<pair<int, int>> contacting_pairs;
        for (int i=0; i<contact_map.size(); i++){
            for (int j=i+1; j<contact_map[i].size(); j++){
                if (contact_map[i][j] && (i-j<=-minimum_distance_for_nontrivial_contact || i-j>=minimum_distance_for_nontrivial_contact)){
                    pair<int, int> temp(i, j);
                    contacting_pairs.push_back(temp);
                }
            }
        }

        for (string node : tree.nodes()){
            if (edge_or_cherry == "edge") {
                string node_seq = msa[node];
                for (adj_pair_t& edge : tree.children(node)){
                    string child = edge.node;
                    float branch_length = edge.length;
                    string child_seq = msa[child];
                    int q_idx = quantization_idx(branch_length, quantization_points);
                    if (q_idx != -1){
                        for (pair<int, int>& p : contacting_pairs){
                            int i = p.first;
                            int j = p.second;
                            string start_state = string{node_seq[i], node_seq[j]};
                            string end_state = string{child_seq[i], child_seq[j]};
                            if (
                                find(amino_acids.begin(), amino_acids.end(), string{node_seq[i]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{node_seq[j]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{child_seq[i]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{child_seq[j]}) != amino_acids.end()
                            ){
                                count_matrices_data[q_idx][aa_pair_to_int[start_state]][aa_pair_to_int[end_state]] += 0.5;
                                reverse(start_state.begin(), start_state.end());
                                reverse(end_state.begin(), end_state.end());
                                count_matrices_data[q_idx][aa_pair_to_int[start_state]][aa_pair_to_int[end_state]] += 0.5;
                            }
                        }
                    }
                }
            } else { // cherry
                vector<adj_pair_t> children = tree.children(node);
                if (children.size() == 2 && all_children_are_leafs(tree, children)){
                    string leaf_1 = children[0].node;
                    float branch_length_1 = children[0].length;
                    string leaf_2 = children[1].node;
                    float branch_length_2 = children[0].length;
                    string leaf_seq_1 = msa[leaf_1];
                    string leaf_seq_2 = msa[leaf_2];
                    float branch_length_total = branch_length_1 + branch_length_2;
                    int q_idx = quantization_idx(branch_length_total, quantization_points);
                    if (q_idx != -1){
                        for (pair<int, int>& p : contacting_pairs){
                            int i = p.first;
                            int j = p.second;
                            string start_state = string{leaf_seq_1[i], leaf_seq_1[j]};
                            string end_state = string{leaf_seq_2[i], leaf_seq_2[j]};
                            if (
                                find(amino_acids.begin(), amino_acids.end(), string{leaf_seq_1[i]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{leaf_seq_1[j]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{leaf_seq_2[i]}) != amino_acids.end()
                                && find(amino_acids.begin(), amino_acids.end(), string{leaf_seq_2[j]}) != amino_acids.end()
                            ){
                                count_matrices_data[q_idx][aa_pair_to_int[start_state]][aa_pair_to_int[end_state]] += 0.25;
                                count_matrices_data[q_idx][aa_pair_to_int[end_state]][aa_pair_to_int[start_state]] += 0.25;
                                reverse(start_state.begin(), start_state.end());
                                reverse(end_state.begin(), end_state.end());
                                count_matrices_data[q_idx][aa_pair_to_int[start_state]][aa_pair_to_int[end_state]] += 0.25;
                                count_matrices_data[q_idx][aa_pair_to_int[end_state]][aa_pair_to_int[start_state]] += 0.25;
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i=0; i<quantization_points.size(); i++){
        count_matrices.push_back(count_matrix{quantization_points[i], count_matrices_data[i]});
    }
    return count_matrices;
}

void write_count_matrices(const vector<count_matrix> & count_matrices, const string & output_count_matrices_dir){
    std::ofstream myfile;
    myfile.open (output_count_matrices_dir);
    int num_matrices = count_matrices.size();
    int num_states = pairs_of_amino_acids.size();
    myfile << num_matrices << " matrices\n" << num_states << " states\n";
    for (const count_matrix& cm : count_matrices){
        myfile << cm.q << "\n";
        myfile << "\t";
        for (string& pair : pairs_of_amino_acids){
            myfile << pair << "\t";
        }
        myfile << "\n";
        for (int i=0; i<num_states; i++){
            myfile << pairs_of_amino_acids[i] << "\t";
            for (int j=0; j<num_states; j++){
                myfile << cm.matrix[i][j];
                if (j != num_states-1){
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
    const vector<string> & families,
    const vector<string> & amino_acids,
    vector<float>& quantization_points,
    const string & edge_or_cherry,
    int minimum_distance_for_nontrivial_contact,
    const string & output_count_matrices_dir,
    int num_processes){

    sort(quantization_points.begin(), quantization_points.end());
    for (const string& a1 : amino_acids){
        for (const string& a2 : amino_acids){
            pairs_of_amino_acids.push_back(a1+a2);
        }
    }
    for (int i=0; i<pairs_of_amino_acids.size(); i++){
        aa_pair_to_int[pairs_of_amino_acids[i]] = i;
    }
    
    vector<count_matrix> count_matrices = _map_func(tree_dir, msa_dir, contact_map_dir, families, 
            amino_acids, quantization_points, edge_or_cherry, minimum_distance_for_nontrivial_contact);
    write_count_matrices(count_matrices, output_count_matrices_dir + "result.txt");
}

int main(int argc, char *argv[]) {
    // Read in all the arguments
    string tree_dir = argv[1];
    string msa_dir = argv[2];
    string contact_map_dir = argv[3];
    int num_of_families = atoi(argv[4]);
    num_of_amino_acids = atoi(argv[5]);
    int num_of_quantization_points = atoi(argv[6]);
    vector<string> families;
    families.reserve(num_of_families);
    for (int i = 0; i < num_of_families; i++) {
        families.push_back(argv[7 + i]);
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
}
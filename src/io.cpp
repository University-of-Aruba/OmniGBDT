#include "io.h"
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <set>
#include <cstring>
#include <sstream>

#define LEFTSTRIP 0
#define RIGHTSTRIP 1
#define BOTHSTRIP 2
#define PRECISION 16

std::string do_strip(const std::string &str, int striptype, const std::string &chars) {
    int len = (int) str.size(), i, j, charslen = (int) chars.size();
    if (charslen == 0) {
        i = 0;
        if (striptype != RIGHTSTRIP) {
            while (i < len && ::isspace(str[i])) {
                i++;
            }
        }

        j = len;
        if (striptype != LEFTSTRIP) {
            do {
                j--;
            } while (j >= i && ::isspace(str[j]));
            j++;
        }
    } else {
        const char *sep = chars.c_str();
        i = 0;
        if (striptype != RIGHTSTRIP) {
            while (i < len && memchr(sep, str[i], charslen)) {
                i++;
            }
        }

        j = len;
        if (striptype != LEFTSTRIP) {
            do {
                j--;
            } while (j >= i && memchr(sep, str[j], charslen));
            j++;
        }
    }

    if (i == 0 && j == len) {
        return str;
    } else {
        return str.substr(i, j - i);
    }
}

std::string strip(const std::string &str, const std::string &chars) {
    return do_strip(str, BOTHSTRIP, chars);
}

std::string lstrip(const std::string &str, const std::string &chars) {
    return do_strip(str, LEFTSTRIP, chars);
}

std::string rstrip(const std::string &str, const std::string &chars) {
    return do_strip(str, RIGHTSTRIP, chars);
}

void split_whitespace(const std::string &str, std::vector<std::string> &result, int maxsplit) {
    std::string::size_type i, j, len = str.size();
    for (i = j = 0; i < len;) {
        while (i < len && ::isspace(str[i])) i++;
        j = i;

        while (i < len && !::isspace(str[i])) i++;

        if (j < i) {
            if (maxsplit-- <= 0) break;
            result.push_back(str.substr(j, i - j));
            while (i < len && ::isspace(str[i])) i++;
            j = i;
        }
    }
    if (j < len) {
        result.push_back(str.substr(j, len - j));
    }
}

void split(const std::string &str, std::vector<std::string> &result, const std::string &sep, int maxsplit) {
    result.clear();
    if (maxsplit < 0) maxsplit = INT_MAX;//result.max_size();
    if (sep.size() == 0) {
        split_whitespace(str, result, maxsplit);
        return;
    }

    std::string::size_type i, j, len = str.size(), n = sep.size();
    i = j = 0;
    while (i + n <= len) {
        if (str[i] == sep[0] && str.substr(i, n) == sep) {
            if (maxsplit-- <= 0) break;
            result.push_back(str.substr(j, i - j));
            i = j = i + n;
        } else {
            i++;
        }
    }
    result.push_back(str.substr(j, len - j));
}

std::string zfill(const std::string &str, int width) {
    int len = (int) str.size();
    if (len >= width) {
        return str;
    }
    std::string s(str);
    int fill = width - len;
    s = std::string(fill, '0') + s;

    if (s[fill] == '+' || s[fill] == '-') {
        s[0] = s[fill];
        s[fill] = '0';
    }

    return s;
}

//////////////////////////////////////////////////////////////////////////////////////////////

int DumpTrees(vector<Tree> &trees, const vector<double> &base_scores, const char *path) {
    ofstream outfile;
    outfile.open(path);
    if (!outfile) { return errno ? errno : EIO; }
    if (!base_scores.empty()) {
        outfile << "BaseScore:";
        for (size_t i = 0; i < base_scores.size(); ++i) {
            if (i > 0) {
                outfile << ",";
            }
            outfile << scientific << setprecision(PRECISION) << base_scores[i];
        }
        outfile << "\n";
    }
    size_t t = 0;
    for (auto &tree : trees) {
        outfile << "Booster[" << t << "]:\n";
        for (auto it : tree.nonleaf) {
            auto v = it.second;
            outfile << "\t" << it.first << "," << v.parent << "," << v.left << "," << v.right << ","
                    << v.column << ",";
            outfile << scientific << setprecision(PRECISION) << v.threshold << endl;
        }

        for (auto &it : tree.leaf) {
            auto v = it.second;
            outfile << "\t\t" << it.first << ",";
            for (int i = 0; i < v.values.size(); i++) {
                if (i < v.values.size() - 1) {
                    outfile << scientific << setprecision(PRECISION) << v.values[i] << ",";
                } else {
                    outfile << scientific << setprecision(PRECISION) << v.values[i] << endl;
                }
            }
        }
        ++t;
    }
    outfile.close();
    return outfile ? 0 : EIO;
}

namespace {

template <typename T>
bool parse_number(const string &field, T &value) {
    istringstream input(field);
    if (!(input >> value)) { return false; }
    input >> ws;
    return input.eof();
}

bool valid_tree(const Tree &tree) {
    if (tree.nonleaf.empty()) { return tree.leaf.size() == 1; }

    set<int> visited;
    vector<int> pending{-1};
    while (!pending.empty()) {
        const int id = pending.back();
        pending.pop_back();
        if (!visited.insert(id).second) { return false; }
        if (id > 0) {
            if (tree.leaf.count(id) == 0) { return false; }
        } else {
            auto node = tree.nonleaf.find(id);
            if (node == tree.nonleaf.end()) { return false; }
            // Older bounded trees use zero for a branch that contributes nothing.
            for (int child : {node->second.left, node->second.right}) {
                if (child != 0) { pending.push_back(child); }
            }
        }
    }
    return visited.size() == tree.nonleaf.size() + tree.leaf.size();
}

}  // namespace

int LoadTrees(vector<Tree> &trees, vector<double> &base_scores, const char *path,
              int inp_dim, int out_dim, int round_width) {
    ifstream infile(path);
    if (!infile) { return errno ? errno : EIO; }

    // The original format wrote a zero-byte file for a zero-round model.
    const bool legacy_empty = infile.peek() == ifstream::traits_type::eof();
    vector<Tree> loaded;
    vector<double> scores;
    vector<string> fields;
    string line;
    while (getline(infile, line)) {
        line = strip(line);
        if (line.empty()) { continue; }
        if (line.compare(0, 10, "BaseScore:") == 0) {
            if (!loaded.empty() || !scores.empty()) { return -1; }
            split(line.substr(10), fields, ",");
            if (fields.size() != static_cast<size_t>(out_dim)) { return -1; }
            scores.resize(out_dim);
            for (int i = 0; i < out_dim; ++i) {
                if (!parse_number(fields[i], scores[i])) { return -1; }
            }
        } else if (line.compare(0, 8, "Booster[") == 0) {
            if (line != "Booster[" + to_string(loaded.size()) + "]:") { return -1; }
            if (!loaded.empty() && !valid_tree(loaded.back())) { return -1; }
            loaded.emplace_back();
        } else {
            if (loaded.empty()) { return -1; }
            Tree &tree = loaded.back();
            split(line, fields, ",");
            int id;
            if (!parse_number(fields[0], id) || id == 0) { return -1; }
            if (id < 0) {
                NonLeafNode node;
                if (fields.size() != 6 || !parse_number(fields[1], node.parent) ||
                    !parse_number(fields[2], node.left) || !parse_number(fields[3], node.right) ||
                    !parse_number(fields[4], node.column) || !parse_number(fields[5], node.threshold) ||
                    node.column < 0 || node.column >= inp_dim ||
                    !tree.nonleaf.emplace(id, node).second) { return -1; }
            } else {
                if (fields.size() - 1 != static_cast<size_t>(out_dim)) { return -1; }
                LeafNode node(out_dim);
                node.parent = 0;  // The text format omits leaf parents.
                for (int i = 0; i < out_dim; ++i) {
                    if (!parse_number(fields[i + 1], node.values[i])) { return -1; }
                }
                if (!tree.leaf.emplace(id, node).second) { return -1; }
            }
        }
    }
    if (infile.bad()) { return EIO; }
    if ((!loaded.empty() && !valid_tree(loaded.back())) ||
        (scores.empty() && loaded.empty() && !legacy_empty) ||
        round_width <= 0 || loaded.size() % round_width != 0) { return -1; }

    // Headerless legacy dumps use the base score configured by the caller.
    if (!scores.empty()) { base_scores.swap(scores); }
    trees.swap(loaded);
    return 0;
}

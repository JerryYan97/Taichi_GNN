#include "../Melting/DirectSolver.h"
#include <Eigen/Eigen>
#include <Math/DIRECT_SOLVER.h>
#include <Math/UTILS.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace JGSL {

py::array_t<float> solve_linear_system_SC(py::array_t<float> input1, py::array_t<float> input2, py::array_t<float> input3, int n, int m) {
    using TStack = Eigen::Matrix<float, Eigen::Dynamic, 1>;

    {
        puts("linear_system :");
        auto M_inv_raw = input1.unchecked<1>();
//        for (ssize_t i = 0; i < n; i++) printf("%.5f ", M_inv_raw(i));
        puts("");
        ssize_t cnt = 0;
        auto G_raw = input2.unchecked<2>();
        for (ssize_t i = 0; i < G_raw.shape(1); i++)
            if (G_raw(2, i) != 0) {
//                printf("%.5f %.5f %.5f\n", G_raw(0, i), G_raw(1, i), G_raw(2, i));
                cnt = std::max(cnt, i);
            }
//        puts("");
//        auto a_raw = input3.unchecked<1>();
//        for (ssize_t i = 0; i < n; i++) printf("%.5f ", a_raw(i));
//        puts("");
//        printf("%d %d\n", n, m);
//        puts("");
        printf("number %d %d %d\n", n, (int)cnt, n);
    }

    TStack M_inv = TStack::Zero(n);
    auto M_inv_raw = input1.unchecked<1>();
    for (ssize_t i = 0; i < n; i++) M_inv(i) = M_inv_raw(i);

    Eigen::SparseMatrix<float> G = Eigen::SparseMatrix<float>(n, m);
    auto G_raw = input2.unchecked<2>();
    std::vector<Eigen::Triplet<float>> G_triplets;
    for (ssize_t i = 0; i < G_raw.shape(1); i++)
        if (G_raw(2, i) != 0)
            G_triplets.emplace_back((int)G_raw(0, i), (int)G_raw(1, i), G_raw(2, i));
    G.setFromTriplets(G_triplets.begin(), G_triplets.end());

    TStack a = TStack::Zero(n);
    auto a_raw = input3.unchecked<1>();
    for (ssize_t i = 0; i < n; i++) a(i) = a_raw(i);

    Eigen::SparseMatrix<float> left = G.transpose() * M_inv.asDiagonal() * G;
    TStack right = G.transpose() * M_inv.asDiagonal() * a;
    TStack solved_p = DirectSolver<float>::solve(left, right, DirectSolver<float>::AMGCL, 1e-4, -1, 100000, -1, 1, false, false);
    TStack solved_v = M_inv.asDiagonal() * a - M_inv.asDiagonal() * G * solved_p;

    auto result = py::array_t<float>(M_inv_raw.shape(0));
    py::buffer_info buf = result.request();
    float* ptr = (float*) buf.ptr;
    for (ssize_t i = 0; i < n; i++) {
        ptr[i] = solved_v(i);
//        printf("!!! %d %.5f\n", i, solved_v(i));
    }
    {
        TStack output = G.transpose() * solved_v.squaredNorm();
        for (int i = 0; i < m; ++i)
            printf("%.5f\n", output(i));
        printf("Divergence : %.20f\n", output.squaredNorm());
    }
//    {
////        M_inv.array() = 1 / M_inv.array();
//        TStack output = G.transpose() * solved_v;
//        for (int i = 0; i < m; ++i)
//            printf("%.5f\n", output(i));
//        printf("Divergence : %.20f\n", output.squaredNorm());
//    }
//    getchar();
    return result;
}

template <class T>
py::array_t<T> solve_linear_system(py::array_t<T> input1, py::array_t<T> input2, int n, py::array_t<int> input3, py::array_t<T> input4, bool verbose) {
    using TStack = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    int dim = 2;
    // compute dirichlet boundary
    std::vector<bool> fixed(n, false);
    auto fb_raw = input3.template unchecked<1>();
    for (int i = 0; i < fb_raw.shape(0); ++i)
        for (int d = 0; d < dim; ++d) fixed[fb_raw(i) * dim + d] = true;
    // compute right
    TStack right = TStack::Zero(n);
    auto right_raw = input2.template unchecked<1>();
    for (int i = 0; i < n; ++i) right(i) = right_raw(i);
    auto x_raw = input4.template unchecked<2>();
    for (int i = 0; i < n; ++i) if (fixed[i]) right[i] = x_raw(i / dim, i % dim);
    // compute left
    Eigen::SparseMatrix<T> left = Eigen::SparseMatrix<T>(n, n);
    auto left_raw = input1.template unchecked<2>();
    std::vector<Eigen::Triplet<T>> left_triplets;
    int tri_cnt = 0;
    for (ssize_t i = 0; i < left_raw.shape(1); i++)
        if (left_raw(2, i) != 0) {
            int r = (int) left_raw(0, i);
            int c = (int) left_raw(1, i);
            T value = left_raw(2, i);
            if (!fixed[r] && !fixed[c]) left_triplets.emplace_back(r, c, value);
            if (!fixed[r] && fixed[c]) right[r] -= x_raw(c / dim, c % dim) * value;
            tri_cnt = i;
        }
    if (tri_cnt == left_raw.shape(1) - 1 || n > right_raw.shape(0)) {
        puts("ERROR, please increase linear system size");
        exit(0);
    }
    if (verbose) printf("\tsystem size : %d %d\n", n, tri_cnt);
    for (int i = 0; i < n; ++i) if (fixed[i]) left_triplets.emplace_back(i, i, 1);
    left.setFromTriplets(left_triplets.begin(), left_triplets.end());
    // compute result
    TStack solved_x = DirectSolver<T>::solve(left, right, DirectSolver<T>::EIGEN, 1e-4, -1, 100000, -1, 1, false, false);
    auto result = py::array_t<T>(right_raw.shape(0));
    py::buffer_info buf = result.request();
    T* ptr = (T*) buf.ptr;
    for (ssize_t i = 0; i < n; i++) ptr[i] = solved_x(i);
    return result;
}

template <class T>
py::array_t<T> solve_linear_system_pd(py::array_t<T> input1, py::array_t<T> input2, int n, py::array_t<int> input3, py::array_t<T> input4, bool verbose, int cc, int total) {
    using TStack = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    int dim = 2;
    // compute dirichlet boundary
    std::vector<bool> fixed(n, false);
    auto fb_raw = input3.template unchecked<1>();
    for (int i = 0; i < fb_raw.shape(0); ++i)
        for (int d = 0; d < dim; ++d) fixed[fb_raw(i) * dim + d] = true;
    // compute right
    TStack right = TStack::Zero(n);
    auto right_raw = input2.template unchecked<1>();
    for (int i = 0; i < n; ++i) right(i) = right_raw(i);
    auto x_raw = input4.template unchecked<2>();
    for (int i = 0; i < n; ++i) if (fixed[i]) right[i] = x_raw(i / dim, i % dim);
    // compute left
    Eigen::SparseMatrix<T> left = Eigen::SparseMatrix<T>(n, n);
    auto left_raw = input1.template unchecked<2>();
    // project pd
    std::vector<Eigen::Triplet<T>> tmp_triplets;
    for (ssize_t i = 0; i < total - 36 * cc; i++)
        tmp_triplets.emplace_back((int)left_raw(0, i), (int)left_raw(1, i), left_raw(2, i));
    for (ssize_t i = 0; i < cc; i++) {
        Eigen::Matrix<T, 6, 6> H;
        int start = total - 36 * cc + 36 * i;
        for (int p = 0; p < 6; ++p)
            for (int q = 0; q < 6; ++q)
                H(p, q) = left_raw(2, start + p * 6 + q);
        makePD(H);
        for (int p = 0; p < 6; ++p)
            for (int q = 0; q < 6; ++q) {
                int idx = start + p * 6 + q;
                tmp_triplets.emplace_back((int)left_raw(0, idx), (int)left_raw(1, idx), H(p, q));
            }
    }
    std::vector<Eigen::Triplet<T>> left_triplets;
    for (ssize_t i = 0; i < total; i++) {
        int r = (int) tmp_triplets[i].row();
        int c = (int) tmp_triplets[i].col();
        T value = tmp_triplets[i].value();
        if (!fixed[r] && !fixed[c]) left_triplets.emplace_back(r, c, value);
        if (!fixed[r] && fixed[c]) right[r] -= x_raw(c / dim, c % dim) * value;
    }
    if (total > left_raw.shape(1) - 1 || n > right_raw.shape(0)) {
        puts("ERROR, please increase linear system size");
        exit(0);
    }
    if (verbose) printf("\tsystem size : %d %d\n", n, total);
    for (int i = 0; i < n; ++i) if (fixed[i]) left_triplets.emplace_back(i, i, 1);
    left.setFromTriplets(left_triplets.begin(), left_triplets.end());
    // compute result
    TStack solved_x = DirectSolver<T>::solve(left, right, DirectSolver<T>::EIGEN, 1e-4, -1, 100000, -1, 1, false, false);
    auto result = py::array_t<T>(right_raw.shape(0));
    py::buffer_info buf = result.request();
    T* ptr = (T*) buf.ptr;
    for (ssize_t i = 0; i < right_raw.shape(0); i++) ptr[i] = 0;
    for (ssize_t i = 0; i < n; i++) ptr[i] = solved_x(i);
    return result;
}

PYBIND11_MODULE(JGSL_WATER, m) {
    m.def("solve_linear_system", &solve_linear_system_pd<float>);
    m.def("solve_linear_system", &solve_linear_system_pd<double>);
}

}
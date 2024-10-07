#pragma once

#include <utility>
#include <vector>

#include <mitsuba/core/fstream.h>

#include <mitsuba/core/logger.h>
#include <mitsuba/mitsuba.h>

#include <drjit/dynamic.h>
#include <drjit/tensor.h>

#include "skymodeldata.h"

NAMESPACE_BEGIN(mitsuba)

    #define NB_TURBIDITY 10
    #define NB_CTRL_PT   6
    #define NB_PARAMS    9

    #define F_DIM 5
    #define L_DIM 4

    enum DataSetShapeIdx { WAVELENGTH = 0, ALBEDO, TURBIDITY, CTRL_PT, PARAMS };

    constexpr size_t f_spec_shape[F_DIM] = {11, 2, NB_TURBIDITY, NB_CTRL_PT, NB_PARAMS};
    constexpr size_t l_spec_shape[L_DIM] = {11, 2, NB_TURBIDITY, NB_CTRL_PT};

    constexpr size_t f_tri_shape[F_DIM] = {3, 2, NB_TURBIDITY, NB_CTRL_PT, NB_PARAMS};
    constexpr size_t l_tri_shape[L_DIM] = {3, 2, NB_TURBIDITY, NB_CTRL_PT};

    struct Dataset {
        size_t nb_dims;
        const size_t* dim_size;

        double** dataset;
    };

    constexpr Dataset f_spectral = {
        .nb_dims = F_DIM,
        .dim_size = f_spec_shape,
        .dataset = datasets
    };

    constexpr Dataset l_spectral = {
        .nb_dims = L_DIM,
        .dim_size = l_spec_shape,
        .dataset = datasetsRad
    };

    constexpr Dataset f_RGB = {
        .nb_dims = F_DIM,
        .dim_size = f_tri_shape,
        .dataset = datasetsRGB
    };

    constexpr Dataset l_RGB = {
        .nb_dims = L_DIM,
        .dim_size = l_tri_shape,
        .dataset = datasetsRGBRad
    };


    void write_tensor_data_v1(const std::string &path, const Dataset& dataset) {
        const auto [nb_dims, dim_size, p_dataset] = dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t)1);

        // Write tensor dimensions
        file.write(nb_dims);

        size_t tensor_size = 1;
        for (size_t dim = 0; dim < nb_dims; ++dim)
            tensor_size *= dim_size[dim];


        // Write reordered shapes
        file.write(dim_size[ALBEDO]);
        file.write(dim_size[TURBIDITY]);

        if (nb_dims == F_DIM) {
            file.write(dim_size[CTRL_PT]);
            file.write(dim_size[WAVELENGTH]);
            file.write(dim_size[PARAMS]);
        } else if (nb_dims == L_DIM) {
            file.write(dim_size[CTRL_PT]);
            file.write(dim_size[WAVELENGTH]);
        } else {
            Throw("Incorrect number of dimensions");
        }

        const size_t nb_params = nb_dims == F_DIM ? NB_PARAMS : 1,
                     nb_colors = dim_size[WAVELENGTH];

        double* buffer = (double*)calloc(tensor_size, sizeof(double));

        // Converts from (11 x 2 x 10 x 6 x ...) to (2 x 10 x 6 x 11 x ...)
        for (size_t a = 0; a < 2; ++a) {
            size_t dest_a_offset = a * (NB_TURBIDITY * NB_CTRL_PT * nb_colors * nb_params),
                   src_a_offset  = a * (NB_TURBIDITY * NB_CTRL_PT * nb_params);

            for (size_t t = 0; t < NB_TURBIDITY; ++t) {
                size_t dest_t_offset = t * (NB_CTRL_PT * nb_colors * nb_params),
                       src_t_offset  = t * NB_CTRL_PT * nb_params;

                for (size_t ctrl_idx = 0; ctrl_idx < NB_CTRL_PT; ++ctrl_idx) {
                    size_t dest_ctrl_offset = ctrl_idx * nb_colors * nb_params,
                           src_ctrl_offset  = ctrl_idx * nb_params;

                    for (size_t color_idx = 0; color_idx < nb_colors; ++color_idx) {
                        size_t dest_col_offset = color_idx * nb_params;

                        for (size_t param_idx = 0; param_idx < nb_params; ++param_idx) {
                            size_t dest_global_offset = dest_a_offset + dest_t_offset + dest_ctrl_offset + dest_col_offset + param_idx,
                                   src_global_offset  = src_a_offset + src_t_offset + src_ctrl_offset + param_idx;

                            buffer[dest_global_offset] = p_dataset[color_idx][src_global_offset];

                        }
                    }
                }
            }
        }

        // Write the data from the dataset
        file.write_array(buffer, tensor_size);

        free(buffer);
        file.close();
    }

    void write_sky_model_data_v1(const std::string &path) {
        write_tensor_data_v1(path + "_v1_spec.bin", f_spectral);
        write_tensor_data_v1(path + "_v1_spec_rad.bin", l_spectral);
        write_tensor_data_v1(path + "_v1_rgb.bin", f_RGB);
        write_tensor_data_v1(path + "_v1_rgb_rad.bin", l_RGB);
    }

    void write_tensor_data_v0(const std::string &path, const Dataset& dataset) {
        const auto [nb_dims, dim_size, p_dataset] = dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t)0);

        // Write tensor dimensions
        file.write(nb_dims);
        size_t lane_size = 1;
        for (size_t dim = 0; dim < nb_dims; ++dim) {
            file.write(dim_size[dim]);
            lane_size *= dim ? dim_size[dim] : 1;
        }

        // Write the data from the dataset
        for (size_t lane_idx = 0; lane_idx < dim_size[0]; ++lane_idx)
            file.write_array(p_dataset[lane_idx], lane_size);

        file.close();
    }

    void write_sky_model_data_v0(const std::string &path) {
        write_tensor_data_v0(path + "_v0_spec.bin", f_spectral);
        write_tensor_data_v0(path + "_v0_spec_rad.bin", l_spectral);
        write_tensor_data_v0(path + "_v0_rgb.bin", f_RGB);
        write_tensor_data_v0(path + "_v0_rgb_rad.bin", l_RGB);
    }

    template<typename Float>
    auto array_from_file(const std::string &path) {
        using FloatStorage  = DynamicBuffer<Float>;
        using DoubleStorage = dr::float64_array_t<FloatStorage>;

        FileStream file(path, FileStream::EMode::ERead);

        // =============== Read headers ===============
        char text_buff[5] = "";
        file.read(text_buff, 3);
        if (strcmp(text_buff, "SKY"))
            Throw("OUPSSS wrong file");

        // Read version
        uint32_t version;
        file.read(version);

        // =============== Read tensor dimensions ===============
        size_t nb_dims = 0;
        file.read(nb_dims);

        size_t nb_elements = 1;
        std::vector<size_t> shape(nb_dims);
        for (size_t dim = 0; dim < nb_dims; ++dim) {
            file.read(shape[dim]);

            if (!shape[dim])
                Throw("Got dimension with 0 elements");

            nb_elements *= shape[dim];
        }

        // ==================== Read data ====================
        double* buffer = static_cast<double *>(
            calloc(nb_elements, sizeof(double)));

        file.read_array(buffer, nb_elements);

        DoubleStorage data_d = dr::load<DoubleStorage>(buffer, nb_elements);
        FloatStorage data_v = FloatStorage(data_d);

        file.close();
        free(buffer);

        return std::pair<std::vector<size_t>, FloatStorage>(shape, data_v);
    }


    template<typename Float>
    auto tensor_from_file(const std::string &path) {
        using FloatStorage  = DynamicBuffer<Float>;
        using DoubleStorage = dr::float64_array_t<FloatStorage>;
        using TensorXf      = dr::Tensor<FloatStorage>;

        FileStream file(path, FileStream::EMode::ERead);

        // =============== Read headers ===============
        char text_buff[5] = "";
        file.read(text_buff, 3);
        if (strcmp(text_buff, "SKY"))
            Throw("OUPSSS wrong file");

        // Read version
        uint32_t version;
        file.read(version);

        // =============== Read tensor dimensions ===============
        size_t nb_dims = 0;
        file.read(nb_dims);

        size_t nb_elements = 1;
        size_t shape[nb_dims];
        for (size_t dim = 0; dim < nb_dims; ++dim) {
            file.read(shape[dim]);

            if (!shape[dim])
                Throw("Got dimension with 0 elements");

            nb_elements *= shape[dim];
        }

        // ==================== Read data ====================
        double* buffer = static_cast<double *>(
            calloc(nb_elements, sizeof(double)));

        file.read_array(buffer, nb_elements);

        DoubleStorage data_d = dr::load<DoubleStorage>(buffer, nb_elements);
        FloatStorage data_v = FloatStorage(data_d);

        file.close();
        free(buffer);

        return TensorXf(data_v, nb_dims, shape);
    }

NAMESPACE_END(mitsuba)

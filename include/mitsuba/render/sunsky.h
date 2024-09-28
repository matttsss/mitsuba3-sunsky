#pragma once

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

    enum DataSetShapeIdx { WAVELENGTH = 0, ALBEDO, TURBIDITY, PARAMS, CTRL_PT };

    constexpr size_t f_spec_shape[F_DIM] = {11, 2, NB_TURBIDITY, NB_PARAMS, NB_CTRL_PT};
    constexpr size_t l_spec_shape[L_DIM] = {11, 2, NB_TURBIDITY, NB_CTRL_PT};

    constexpr size_t f_tri_shape[F_DIM] = {3, 2, NB_TURBIDITY, NB_PARAMS, NB_CTRL_PT};
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

    constexpr Dataset f_XYZ = {
        .nb_dims = F_DIM,
        .dim_size = f_tri_shape,
        .dataset = datasetsXYZ
    };

    constexpr Dataset l_XYZ = {
        .nb_dims = L_DIM,
        .dim_size = l_tri_shape,
        .dataset = datasetsXYZRad
    };


    void write_tensor_data_v2(const std::string &path, const Dataset& dataset) {
        const auto [nb_dims, dim_size, p_dataset] = dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t)2);

        // Write tensor dimensions
        file.write(nb_dims);

        size_t tensor_size = 1;
        for (size_t dim = 0; dim < nb_dims; ++dim)
            tensor_size *= dim_size[dim];


        double* buffer = (double*) calloc(tensor_size, sizeof(double));


        // Write reordered shapes
        file.write(dim_size[ALBEDO]); // albedo
        file.write(dim_size[TURBIDITY]); // turbidity
        if (nb_dims == L_DIM) {
            file.write(dim_size[CTRL_PT-1]); // control points
        } else if (nb_dims == F_DIM) {
            file.write(dim_size[CTRL_PT]);
            file.write(dim_size[PARAMS]);
        } else {
            Throw("Tensor has incompatible dim");
        }

        file.write(dim_size[WAVELENGTH]);


        const size_t nb_param = nb_dims == F_DIM ? NB_PARAMS : 1,
                     nb_colors = dim_size[WAVELENGTH];

        // Converts from (11 x 2 x 10 x ... x 6) to (2 x 10 x ... x 6 x 11)
        // Let the cache butchering start
        for (size_t a = 0; a < 2; ++a) {
            size_t a_offset = a * (NB_TURBIDITY * NB_CTRL_PT * nb_param * nb_colors);

            for (size_t t = 0; t < NB_TURBIDITY; ++t) {
                size_t t_offset = t * (NB_CTRL_PT * nb_param * nb_colors);

                for (size_t ctrl_idx = 0; ctrl_idx < NB_CTRL_PT; ++ctrl_idx) {
                    size_t ctrl_offset = ctrl_idx * (nb_param * nb_colors);

                    for (size_t param_idx = 0; param_idx < nb_param; ++param_idx) {
                        size_t param_offset = param_idx * nb_colors;

                        for (size_t lbda = 0; lbda < nb_colors; ++lbda) {
                            size_t global_offset = a_offset + t_offset + ctrl_offset + param_offset + lbda;

                            // Thought it was 11 x 10 x 2 x ... when it was 11 x 2 x 10 x ... :(
                            //buffer[global_idx] = p_dataset[lbda][t * (2 * nb_param * NB_CTRL_PT) +
                            //                                     a * (nb_param * NB_CTRL_PT) + ctrl_idx];

                            buffer[global_offset] = p_dataset[lbda][a * (10 * nb_param * NB_CTRL_PT) +
                                                                    t * (nb_param * NB_CTRL_PT) + ctrl_idx];

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

    void write_sky_model_data_v2(const std::string &path) {
        write_tensor_data_v2(path + "_v2_spec.bin", f_spectral);
        write_tensor_data_v2(path + "_v2_spec_rad.bin", l_spectral);
        write_tensor_data_v2(path + "_v2_rgb.bin", f_RGB);
        write_tensor_data_v2(path + "_v2_rgb_rad.bin", l_RGB);
        write_tensor_data_v2(path + "_v2_xyz.bin", f_XYZ);
        write_tensor_data_v2(path + "_v2_xyz_rad.bin", l_XYZ);
    }

    void write_tensor_data_v3(const std::string &path, const Dataset& dataset) {
        const auto [nb_dims, dim_size, p_dataset] = dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t)3);

        // Write tensor dimensions
        file.write(nb_dims);

        size_t tensor_size = 1;
        for (size_t dim = 0; dim < nb_dims; ++dim)
            tensor_size *= dim_size[dim];


        double* buffer = (double*) calloc(tensor_size, sizeof(double));

        // Write reordered shapes
        file.write(dim_size[ALBEDO]); // albedo
        file.write(dim_size[TURBIDITY]); // turbidity
        if (nb_dims == L_DIM) {
            file.write(dim_size[WAVELENGTH]);
            file.write(dim_size[CTRL_PT-1]); // control points since it has one dim less
        } else if (nb_dims == F_DIM) {
            file.write(dim_size[PARAMS]);
            file.write(dim_size[WAVELENGTH]);
            file.write(dim_size[CTRL_PT]);
        } else {
            file.close();
            Throw("Tensor has incompatible dim");
        }


        const size_t nb_param = nb_dims == F_DIM ? NB_PARAMS : 1,
                     nb_colors = dim_size[WAVELENGTH];

        // Converts from (11 x 2 x 10 x ... x 6) to (2 x 10 x ... x 11 x 6)
        // Let the cache butchering start
        for (size_t a = 0; a < 2; ++a) {
            size_t a_offset = a * (NB_TURBIDITY * NB_CTRL_PT * nb_param * nb_colors);

            for (size_t t = 0; t < NB_TURBIDITY; ++t) {
                size_t t_offset = t * (NB_CTRL_PT * nb_param * nb_colors);

                for (size_t param_idx = 0; param_idx < nb_param; ++param_idx) {
                    size_t param_offset = param_idx * NB_CTRL_PT * nb_colors;

                    for (size_t color_idx = 0; color_idx < nb_colors; ++color_idx) {
                        size_t dest_global_offset = a_offset + t_offset + param_offset + color_idx * NB_CTRL_PT,
                               src_global_offset  = a * (NB_TURBIDITY * NB_CTRL_PT * nb_param) + t * (nb_param * NB_CTRL_PT) + param_idx * NB_CTRL_PT;

                        memcpy(buffer + dest_global_offset,
                               p_dataset[color_idx] + src_global_offset,
                               NB_CTRL_PT * sizeof(double));

                    }

                }

            }

        }

        // Write the data from the dataset
        file.write_array(buffer, tensor_size);

        free(buffer);
        file.close();
    }

    void write_sky_model_data_v3(const std::string &path) {
        write_tensor_data_v3(path + "_v3_spec.bin", f_spectral);
        write_tensor_data_v3(path + "_v3_spec_rad.bin", l_spectral);
        write_tensor_data_v3(path + "_v3_rgb.bin", f_RGB);
        write_tensor_data_v3(path + "_v3_rgb_rad.bin", l_RGB);
        write_tensor_data_v3(path + "_v3_xyz.bin", f_XYZ);
        write_tensor_data_v3(path + "_v3_xyz_rad.bin", l_XYZ);
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
        write_tensor_data_v0(path + "_v0_xyz.bin", f_XYZ);
        write_tensor_data_v0(path + "_v0_xyz_rad.bin", l_XYZ);
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

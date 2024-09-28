#pragma once

#include <mitsuba/core/fstream.h>

#include <mitsuba/core/logger.h>
#include <mitsuba/mitsuba.h>

#include <drjit/dynamic.h>
#include <drjit/tensor.h>

#include "skymodeldata.h"

NAMESPACE_BEGIN(mitsuba)

    #define NB_TURBIDITY 10
    #define NB_CTRL_PT      6
    #define NB_PARAMS       9

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
        write_tensor_data_v2(path + ".bin", f_spectral);
        write_tensor_data_v2(path + ".rad.bin", l_spectral);
    }

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
            tensor_size *=  dim_size[dim];


        // Write reordered shapes
        file.write(dim_size[2]); // albedo
        file.write(dim_size[1]); // turbidity
        file.write(dim_size[0]); // color chanels
        for (size_t i = 3; i < nb_dims; ++i)
            file.write(dim_size[i]);


        // data_count = nb of elements once color, albedo and turbidity are set
        size_t data_count = dataset.nb_dims == 4 ?
                dim_size[3] :
                dim_size[3]*dim_size[4];


        size_t nb_colors = dim_size[0];
        double* buffer = (double*)calloc(tensor_size, sizeof(double));

        // Converts from (11 x 10 x 2 x ... x 6) to (2 x 10 x 11 x ... x 6)
        for (size_t a = 0; a<2; ++a) {
            for (size_t t = 0; t<10; ++t) {
                for (size_t lbda = 0; lbda<nb_colors; ++lbda) {
                    size_t mem_offset = a * (10 * nb_colors * data_count) +
                                        t * (nb_colors * data_count) +
                                        lbda * data_count;

                    memcpy(buffer + mem_offset,
                           p_dataset[lbda] + a*(10*data_count) + t*data_count,
                           data_count*sizeof(double));
                }
            }
        }

        // Write the data from the dataset
        file.write_array(buffer, tensor_size);

        free(buffer);
        file.close();
    }

    void write_sky_model_data_v1(const std::string &path) {
        write_tensor_data_v1(path + ".bin", f_spectral);
        write_tensor_data_v1(path + ".rad.bin", l_spectral);
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
        write_tensor_data_v0(path + ".bin", f_spectral);
        write_tensor_data_v0(path + ".rad.bin", l_spectral);
    }




    template<typename Float>
    auto tensor_from_file(const std::string &path) {
        using FloatStorage  = DynamicBuffer<Float>;
        using DoubleStorage = dr::float64_array_t<FloatStorage>;
        //using DoubleStorage = dr::float64_array_t<double>;
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

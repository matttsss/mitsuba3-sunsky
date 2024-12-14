#pragma once

#include <utility>
#include <vector>

#include "sunsky_helpers.h"
#include "ArHosekSkyModelData_Spectral.h"
#include "ArHosekSkyModelData_RGB.h"
#include "ArHosekSkyModelData_CIEXYZ.h"
#include <mitsuba/core/fstream.h>

#include <drjit/dynamic.h>
#include <drjit/tensor.h>

NAMESPACE_BEGIN(mitsuba)

    // ================================================================================================
    // =========================================== FILE I/O ===========================================
    // ================================================================================================

    #define F_DIM 5
    #define L_DIM 4

    enum class SunDataShapeIdx: uint32_t { WAVELENGTH = 0, TURBIDITY, SUN_SEGMENTS, SUN_CTRL_PTS };
    enum class SkyDataShapeIdx: uint32_t { WAVELENGTH = 0, ALBEDO, TURBIDITY, CTRL_PT, PARAMS };

    constexpr size_t solar_shape[4] = {NB_WAVELENGTHS, NB_TURBIDITY, NB_SUN_SEGMENTS, NB_SUN_CTRL_PTS};
    constexpr size_t limb_darkening_shape[2] = {NB_WAVELENGTHS, 6};

    constexpr size_t f_spec_shape[F_DIM] = {NB_WAVELENGTHS, NB_ALBEDO, NB_TURBIDITY, NB_SKY_CTRL_PTS, NB_SKY_PARAMS};
    constexpr size_t l_spec_shape[L_DIM] = {NB_WAVELENGTHS, NB_ALBEDO, NB_TURBIDITY, NB_SKY_CTRL_PTS};

    constexpr size_t f_tri_shape[F_DIM] = {3, NB_ALBEDO, NB_TURBIDITY, NB_SKY_CTRL_PTS, NB_SKY_PARAMS};
    constexpr size_t l_tri_shape[L_DIM] = {3, NB_ALBEDO, NB_TURBIDITY, NB_SKY_CTRL_PTS};

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

    constexpr Dataset solar_dataset = {
        .nb_dims = 4,
        .dim_size = solar_shape,
        .dataset = solarDatasets
    };

    constexpr Dataset limb_darkening_dataset = {
        .nb_dims = 2,
        .dim_size = limb_darkening_shape,
        .dataset = limbDarkeningDatasets
    };

    void write_limb_darkening_data(const std::string& path) {
        const auto [nb_dims, dim_size, p_dataset] = limb_darkening_dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SUN", 3);
        file.write((uint32_t)0);

        // Write tensor dimensions
        file.write(nb_dims);

        // Write reordered shapes
        // as [wavelengths x nb_params]
        file.write(dim_size[0]);
        file.write(dim_size[1]);

        for (size_t w = 0; w < NB_WAVELENGTHS; ++w)
            file.write(p_dataset[w], 6 * sizeof(double));

        file.close();
    }

    void write_sun_data(const std::string& path) {
        const auto [nb_dims, dim_size, p_dataset] = solar_dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SUN", 3);
        file.write((uint32_t)0);

        // Write tensor dimensions
        file.write(nb_dims);

        // Write reordered shapes
        file.write(dim_size[(uint32_t)SunDataShapeIdx::TURBIDITY]);
        file.write(dim_size[(uint32_t)SunDataShapeIdx::WAVELENGTH]);
        file.write(dim_size[(uint32_t)SunDataShapeIdx::SUN_SEGMENTS]);
        file.write(dim_size[(uint32_t)SunDataShapeIdx::SUN_CTRL_PTS]);

        for (size_t turb = 0; turb < NB_TURBIDITY; ++turb) {
            for (size_t lambda = 0; lambda < NB_WAVELENGTHS; ++lambda) {
                for (size_t segment = 0; segment < NB_SUN_SEGMENTS; ++segment) {
                    for (size_t ctrl_pt = 0; ctrl_pt < NB_SUN_CTRL_PTS; ++ctrl_pt) {
                        // Weird indices since their dataset goes backwards on the last index
                        const size_t src_global_offset = turb * (NB_SUN_SEGMENTS * NB_SUN_CTRL_PTS) +
                                                         (segment + 1) * NB_SUN_CTRL_PTS - (ctrl_pt + 1);

                        file.write(p_dataset[lambda][src_global_offset]);
                    }
                }
            }
        }

        file.close();
    }


    void write_sky_data(const std::string &path, const Dataset& dataset) {
        const auto [nb_dims, dim_size, p_dataset] = dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t)0);

        // Write tensor dimensions
        file.write(nb_dims);

        size_t tensor_size = 1;
        for (size_t dim = 0; dim < nb_dims; ++dim)
            tensor_size *= dim_size[dim];


        // Write reordered shapes
        file.write(dim_size[(uint32_t)SkyDataShapeIdx::TURBIDITY]);
        file.write(dim_size[(uint32_t)SkyDataShapeIdx::ALBEDO]);
        file.write(dim_size[(uint32_t)SkyDataShapeIdx::CTRL_PT]);
        file.write(dim_size[(uint32_t)SkyDataShapeIdx::WAVELENGTH]);

        if (nb_dims == F_DIM)
            file.write(dim_size[(uint32_t)SkyDataShapeIdx::PARAMS]);


        const size_t nb_params = nb_dims == F_DIM ? NB_SKY_PARAMS : 1,
                     nb_colors = dim_size[(uint32_t)SkyDataShapeIdx::WAVELENGTH];

        double* buffer = (double*)calloc(tensor_size, sizeof(double));

        // Converts from (11 x 2 x 10 x 6 x ...) to (10 x 2 x 6 x 11 x ...)
        for (size_t t = 0; t < NB_TURBIDITY; ++t) {

            for (size_t a = 0; a < NB_ALBEDO; ++a) {

                for (size_t ctrl_idx = 0; ctrl_idx < NB_SKY_CTRL_PTS; ++ctrl_idx) {

                    for (size_t color_idx = 0; color_idx < nb_colors; ++color_idx) {

                        for (size_t param_idx = 0; param_idx < nb_params; ++param_idx) {
                            size_t dest_global_offset = t * (NB_ALBEDO * NB_SKY_CTRL_PTS * nb_colors * nb_params) +
                                                        a * (NB_SKY_CTRL_PTS * nb_colors * nb_params) +
                                                        ctrl_idx * nb_colors * nb_params +
                                                        color_idx * nb_params +
                                                        param_idx;
                            size_t src_global_offset = a * (NB_TURBIDITY * NB_SKY_CTRL_PTS * nb_params) +
                                                       t * (NB_SKY_CTRL_PTS * nb_params) +
                                                       ctrl_idx * nb_params +
                                                       param_idx;
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

    void write_sun_sky_model_data(const std::string &path) {
        write_sky_data(path + "_spec.bin", f_spectral);
        write_sky_data(path + "_spec_rad.bin", l_spectral);
        write_sky_data(path + "_rgb.bin", f_RGB);
        write_sky_data(path + "_rgb_rad.bin", l_RGB);
        write_sky_data(path + "_xyz.bin", f_XYZ);
        write_sky_data(path + "_xyz_rad.bin", l_XYZ);
        write_sun_data(path + "_solar.bin");
        write_limb_darkening_data(path + "_ld_sun.bin");
    }

    template<typename FileType, typename OutType>
    std::vector<OutType> array_from_file(const std::string &path) {
        FileStream file(path, FileStream::EMode::ERead);

        // =============== Read headers ===============
        char text_buff[5] = "";
        file.read(text_buff, 3);
        if (strcmp(text_buff, "SKY") != 0 && strcmp(text_buff, "SUN") != 0)
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
        std::vector<FileType> data_f(nb_elements);
        file.read_array(data_f.data(), nb_elements);
        file.close();

        return std::vector<OutType>(data_f.begin(), data_f.end());
    }

    template<typename Float>
    void array_to_file(const std::string &path, const DynamicBuffer<Float>& data, const std::vector<size_t>& shape = {}) {
        std::vector<size_t> _shape = shape.empty() ? std::vector<size_t>{data.size()} : shape;
        if (_shape.empty())
            _shape.push_back(data.size());

        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // =============== Write headers ===============
        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t) 0);

        // =============== Write dimensions ===============
        file.write((size_t) _shape.size());

        for (size_t dim_length : _shape) {
            file.write(dim_length);

            if (!dim_length)
                Throw("Got dimension with 0 elements");
        }

        // ==================== Write data ====================
        file.write_array(data.data(), data.size());

        file.close();
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
        if (strcmp(text_buff, "SKY") != 0 && strcmp(text_buff, "SUN") != 0)
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
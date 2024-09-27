#pragma once

#include <mitsuba/core/fstream.h>

#include <mitsuba/core/logger.h>
#include <mitsuba/mitsuba.h>

#include <drjit/dynamic.h>
#include <drjit/tensor.h>

#include "skymodeldata.h"

NAMESPACE_BEGIN(mitsuba)

    #define SIZE_WAVELENGTH_LINE (10 * 2 * 9 * 6)

    void write_sky_model_data(const std::string &path) {
        constexpr size_t NB_DIMS = 5;
        constexpr size_t DEFAULT_DIMS[NB_DIMS] = {11, 10, 2, 9, 6};

        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t)0);

        // Write tensor dimensions
        file.write(&NB_DIMS, sizeof(NB_DIMS));
        for (size_t dim : DEFAULT_DIMS)
            file.write(dim);

        // Write the data from the dataset
        for (const double* dataset : datasets)
            file.write_array(dataset, SIZE_WAVELENGTH_LINE);

        file.close();
    }

    template<typename Float>
    auto read_sky_model_data(const std::string &path) {
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

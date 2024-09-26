#pragma once

#include <mitsuba/core/fstream.h>

#include <mitsuba/core/logger.h>
#include <mitsuba/mitsuba.h>

NAMESPACE_BEGIN(mitsuba)

    void write_sky_model_data(const std::string &path);

    MI_VARIANT
    dr::Tensor<DynamicBuffer<Float>> read_sky_model_data(const std::string &path) {
        using FloatStorage  = DynamicBuffer<Float>;
        using DoubleStorage = dr::float64_array_t<FloatStorage>;
        //using DoubleStorage = dr::float64_array_t<double>;
        using TensorXf      = dr::Tensor<FloatStorage>;

        FileStream file(path, FileStream::EMode::ERead);

        // =============== Read headers ===============
        char text_buff[5] = "";
        file.read(text_buff, 3);
        if (!strcmp(text_buff, "SKY"))
            Throw("OUPSSS wrong file");

        // Read version
        file.read(text_buff, 4);

        // =============== Read tensor dimensions ===============
        uint8_t nb_dims = 0;
        file.read(nb_dims);

        uint32_t nb_elements = 1;
        uint32_t shape[nb_dims];
        for (uint8_t dim = 0; dim < nb_dims; ++dim) {
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

        // TODO sort shapes (+1 necessary)?
        return TensorXf(data_v, nb_dims, shape);
    }

NAMESPACE_END(mitsuba)

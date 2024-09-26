#include "skymodeldata.h"

#include <mitsuba/core/fstream.h>
#include <drjit/tensor.h>

using namespace drjit;

NAMESPACE_BEGIN(mitsuba)

#define SIZE_WAVELENGTH_LINE (10 * 2 * 9 * 6)

    static constexpr uint8_t NB_DIMS = 5;

    static constexpr uint32_t defaults[NB_DIMS] = {11, 10, 2, 9, 6};

    void write_sky_model_data(const fs::path &path) {
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write("v000", 4);

        // Write tensor dimensions
        file.write(&NB_DIMS, sizeof(NB_DIMS));
        for (uint8_t dim = 0; dim < NB_DIMS; ++dim)
            file.write(defaults[dim]);

        // Write the data from the dataset
        for (const double* dataset : datasets)
            file.write_array(dataset, SIZE_WAVELENGTH_LINE);

        file.close();
    }

    MI_VARIANT
    Tensor<DynamicBuffer<Float>> read_sky_model_data(const fs::path &path) {
        using FloatStorage  = DynamicBuffer<Float>;
        using DoubleStorage = float64_array_t<FloatStorage>;
        using TensorXf      = Tensor<FloatStorage>;

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

        uint32_t dim_size = 0,
                 nb_elements = 1;
        for (uint8_t dim = 0; dim < nb_dims; ++dim) {
            file.read(dim_size);

            nb_elements *= dim_size ? dim_size : 1;
        }

        // ==================== Read data ====================
        double* buffer = static_cast<double *>(
            calloc(nb_elements, sizeof(double)));

        file.read_array(buffer, nb_elements);

        DoubleStorage data_d = dr::load<DoubleStorage>(buffer, nb_elements);
        FloatStorage data_v = FloatStorage(data_d);

        file.close();
        free(buffer);

        // TODO sort shapes
        return TensorXf(data_v);
    }

NAMESPACE_END(mitsuba)

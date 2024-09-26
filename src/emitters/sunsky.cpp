#include "skymodeldata.h"

#include <mitsuba/render/sunsky.h>
#include <mitsuba/core/fstream.h>

NAMESPACE_BEGIN(mitsuba)

    #define SIZE_WAVELENGTH_LINE (10 * 2 * 9 * 6)

    void write_sky_model_data(const std::string &path) {
        constexpr uint8_t NB_DIMS = 5;
        constexpr uint32_t DEFAULT_DIMS[NB_DIMS] = {11, 10, 2, 9, 6};

        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write("v000", 4);

        // Write tensor dimensions
        file.write(&NB_DIMS, sizeof(NB_DIMS));
        for (uint32_t dim : DEFAULT_DIMS)
            file.write(dim);

        // Write the data from the dataset
        for (const double* dataset : datasets)
            file.write_array(dataset, SIZE_WAVELENGTH_LINE);

        file.close();
    }

    // TODO Equivalent of "MI_IMPLEMENT_CLASS_VARIANT(Texture, Object, "texture")"
    // or "MI_INSTANTIATE_CLASS(Texture)"
    MI_VARIANT
    dr::Tensor<DynamicBuffer<Float>> dumby_name(const std::string &path) {
        using FloatStorage  = DynamicBuffer<Float>;
        //using DoubleStorage = dr::float64_array_t<FloatStorage>;
        using DoubleStorage = dr::float64_array_t<double>;
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

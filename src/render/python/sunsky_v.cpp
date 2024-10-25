#include <mitsuba/render/sunsky.h>
#include <mitsuba/python/python.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

MI_PY_EXPORT(sunsky) {
    MI_PY_IMPORT_TYPES()

    m.def("write_sky_model_data_v0", &write_sky_model_data_v0, "path"_a)
     .def("write_sky_model_data_v1", &write_sky_model_data_v1, "path"_a)
     .def("write_sky_model_data_v2", &write_sky_model_data_v2, "path"_a)
     .def("tensor_from_file", &tensor_from_file<Float>, "path"_a)
     .def("array_from_file_d", &array_from_file_d<Float>, "path"_a)
     .def("array_from_file_f", &array_from_file_f<Float>, "path"_a)
     .def("array_to_file", &array_to_file<Float>, "path"_a, "data"_a, "shape"_a = std::vector<size_t>());

}

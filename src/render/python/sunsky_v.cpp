#include <nanobind/nanobind.h>
#include <mitsuba/render/sunsky/sunsky.h>

#include <mitsuba/python/python.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

MI_PY_EXPORT(sunsky) {
    MI_PY_IMPORT_TYPES()

    m.def("write_sun_sky_model_data", &write_sun_sky_model_data, "path"_a)
     .def("tensor_from_file", &tensor_from_file<Float>, "path"_a)
     .def("array_from_file_f", &array_from_file<float, ScalarFloat>, "path"_a)
     .def("array_from_file_d", &array_from_file<double, ScalarFloat>, "path"_a)
     .def("array_to_file", &array_to_file<Float>, "path"_a, "data"_a, "shape"_a = std::vector<size_t>());
}

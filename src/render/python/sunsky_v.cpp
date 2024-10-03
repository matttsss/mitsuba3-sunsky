#include <mitsuba/render/sunsky.h>
#include <mitsuba/python/python.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

MI_PY_EXPORT(sunsky) {
    MI_PY_IMPORT_TYPES()

    m.def("write_sky_model_data_v0", &write_sky_model_data_v0, "path"_a)
     .def("write_sky_model_data_v1", &write_sky_model_data_v1, "path"_a)
     .def("tensor_from_file", &tensor_from_file<Float>, "path"_a)
     .def("array_from_file", &array_from_file<Float>, "path"_a);

}

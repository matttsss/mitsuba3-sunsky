#include <mitsuba/render/sunsky.h>
#include <mitsuba/python/python.h>

#include <nanobind/stl/string.h>

MI_PY_EXPORT(sunsky) {
    MI_PY_IMPORT_TYPES()

    m.def("write_sky_model_data_v0", &write_sky_model_data_v0, "path"_a)
     .def("write_sky_model_data_v1", &write_sky_model_data_v1, "path"_a)
     .def("read_sky_model_data", &tensor_from_file<Float>, "path"_a);

}

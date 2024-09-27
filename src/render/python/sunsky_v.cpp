#include <mitsuba/render/sunsky.h>
#include <mitsuba/python/python.h>

#include <nanobind/stl/string.h>

MI_PY_EXPORT(sunsky) {
    MI_PY_IMPORT_TYPES()

    m.def("write_sky_model_data", &write_sky_model_data, "path"_a)
     .def("read_sky_model_data", &read_sky_model_data<Float>, "path"_a);

}

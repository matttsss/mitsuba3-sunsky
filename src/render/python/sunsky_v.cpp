#include <nanobind/nanobind.h>
#include <mitsuba/render/sunsky/sunsky.h>

#include <mitsuba/python/python.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <mitsuba/render/sunsky/ArHosekSkyModel.c>

MI_PY_EXPORT(sunsky) {
    MI_PY_IMPORT_TYPES()

    m.def("write_sun_sky_model_data", &write_sun_sky_model_data, "path"_a)
     .def("array_from_file_f", &array_from_file<Float, Float>, "path"_a)
     .def("array_from_file_d", &array_from_file<Float64, Float>, "path"_a)
     .def("array_to_file", &array_to_file<Float>, "path"_a, "data"_a, "shape"_a = std::vector<size_t>())
     .def("hosek_sun_rad", &arhosekskymodel_solar_radiance_internal2, "turbidity"_a, "wavelength"_a, "elevation"_a, "gamma"_a);
}

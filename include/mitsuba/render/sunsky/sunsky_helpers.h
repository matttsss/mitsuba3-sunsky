#pragma once

#include <drjit/sphere.h>
#include <mitsuba/core/fwd.h>

#include <vector>

#define NB_WAVELENGTHS 11
#define NB_TURBIDITY 10
#define NB_ALBEDO 2

static constexpr size_t WAVELENGTH_STEP = 40;
static constexpr float WAVELENGTHS[NB_WAVELENGTHS] = {
    320, 360, 400, 420, 460, 520, 560, 600, 640, 680, 720
};

#define NB_SKY_CTRL_PTS 6
#define NB_SKY_PARAMS 9

#define NB_SUN_CTRL_PTS 4
#define NB_SUN_SEGMENTS 45
#define NB_SUN_LD_PARAMS 6


NAMESPACE_BEGIN(mitsuba)

    template<typename ScalarFloat>
    std::vector<ScalarFloat> lerp_vectors(const std::vector<ScalarFloat>& a, const std::vector<ScalarFloat>& b, const ScalarFloat& t) {
        assert(a.size() == b.size());

        std::vector<ScalarFloat> res(a.size());
        for (size_t i = 0; i < a.size(); ++i)
            res[i] = dr::lerp(a[i], b[i], t);

        return res;
    }

    template<typename Value>
    Vector<Value, 3> to_spherical(const Point<Value, 2>& angles) {
        auto [sp, cp] = dr::sincos(angles.x());
        auto [st, ct] = dr::sincos(angles.y());

        return {
            cp * st, sp * st, ct
        };
    }

    template<typename Value>
    Point<Value, 2> from_spherical(const Vector<Value, 3>& v) {
        return {
            dr::atan2(v.y(), v.x()),
            dr::unit_angle_z(v)
        };
    }

NAMESPACE_END(mitsuba)
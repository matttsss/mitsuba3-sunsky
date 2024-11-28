#pragma once

#include <vector>

#include <mitsuba/core/fwd.h>

#define NB_TURBIDITY 10
#define NB_ALBEDO 2
#define NB_CTRL_PTS 6
#define NB_PARAMS 9


NAMESPACE_BEGIN(mitsuba)

    template<typename ScalarFloat>
    std::vector<ScalarFloat> lerp_vectors(const std::vector<ScalarFloat>& a, const std::vector<ScalarFloat>& b, const ScalarFloat& t) {
        assert(a.size() == b.size());

        std::vector<ScalarFloat> res(a.size());
        for (size_t i = 0; i < a.size(); ++i)
            res[i] = dr::lerp(a[i], b[i], t);

        return res;
    }

    template <typename ScalarFloat>
    std::vector<ScalarFloat> bezier_interpolate(
        const std::vector<ScalarFloat>& dataset, size_t out_size,
        const uint32_t& offset, const ScalarFloat& x) {

        constexpr ScalarFloat coefs[NB_CTRL_PTS] =
            {1, 5, 10, 10, 5, 1};

        std::vector<ScalarFloat> res(out_size, 0.0f);
        for (size_t i = 0; i < NB_CTRL_PTS; ++i) {
            ScalarFloat coef = coefs[i] * dr::pow(1 - x, 5 - i) * dr::pow(x, i);
            uint32_t index = offset + i * out_size;
            for (size_t j = 0; j < out_size; ++j)
                res[j] += coef * dataset[index + j];
        }

        return res;
    }


    template <typename ScalarFloat, typename Albedo>
    std::vector<ScalarFloat> compute_radiance_params(
        const std::vector<ScalarFloat>& dataset,
        const Albedo& albedo,
        ScalarFloat turbidity, ScalarFloat eta) {

        turbidity = dr::clip(turbidity, 1.f, NB_TURBIDITY);
        eta = dr::clip(eta, 0.f, 0.5f * dr::Pi<ScalarFloat>);

        ScalarFloat x = dr::pow(2 * dr::InvPi<ScalarFloat> * eta, 1.f/3.f);

        uint32_t t_int = dr::floor2int<uint32_t>(turbidity),
                     t_low = dr::maximum(t_int - 1, 0),
                     t_high = dr::minimum(t_low + 1, NB_TURBIDITY - 1);

        ScalarFloat t_rem = turbidity - t_int;

        const size_t t_block_size = dataset.size() / NB_TURBIDITY,
                     a_block_size = t_block_size / NB_ALBEDO,
                     ctrl_block_size = a_block_size / NB_CTRL_PTS,
                     inner_block_size = ctrl_block_size / albedo.size();
                                     // albedo.size() <==> NB_CHANNELS

        std::vector<ScalarFloat>
            t_low_a_low = bezier_interpolate(dataset, ctrl_block_size, t_low * t_block_size + 0 * a_block_size, x),
            t_high_a_low = bezier_interpolate(dataset, ctrl_block_size, t_high * t_block_size + 0 * a_block_size, x),
            t_low_a_high = bezier_interpolate(dataset, ctrl_block_size, t_low * t_block_size + 1 * a_block_size, x),
            t_high_a_high = bezier_interpolate(dataset, ctrl_block_size, t_high * t_block_size + 1 * a_block_size, x);

        std::vector<ScalarFloat>
            res_a_low = lerp_vectors(t_low_a_low, t_high_a_low, t_rem),
            res_a_high = lerp_vectors(t_low_a_high, t_high_a_high, t_rem);

        std::vector<ScalarFloat> res(ctrl_block_size);
        for (size_t i = 0; i < ctrl_block_size; ++i)
            res[i] = dr::lerp(res_a_low[i], res_a_high[i], albedo[i/inner_block_size]);

        return res;
    }

NAMESPACE_END(mitsuba)
#pragma once

#include <vector>
#include "sunsky_helpers.h"

NAMESPACE_BEGIN(mitsuba)

    template <typename ScalarFloat>
    std::vector<ScalarFloat> bezier_interpolate(
        const std::vector<ScalarFloat>& dataset, size_t out_size,
        const uint32_t& offset, const ScalarFloat& x) {

        constexpr ScalarFloat coefs[NB_SKY_CTRL_PTS] =
            {1, 5, 10, 10, 5, 1};

        std::vector<ScalarFloat> res(out_size, 0.0f);
        for (size_t ctrl_pt = 0; ctrl_pt < NB_SKY_CTRL_PTS; ++ctrl_pt) {

            uint32_t index = offset + ctrl_pt * out_size;
            ScalarFloat coef = coefs[ctrl_pt] * dr::pow(1 - x, 5 - ctrl_pt) * dr::pow(x, ctrl_pt);

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
                     a_block_size = t_block_size  / NB_ALBEDO,
                     result_size  = a_block_size / NB_SKY_CTRL_PTS,
                     nb_params    = result_size / albedo.size(); // Either 1 or 9
                                 // albedo.size() <==> NB_CHANNELS

        std::vector<ScalarFloat>
            t_low_a_low = bezier_interpolate(dataset, result_size, t_low * t_block_size + 0 * a_block_size, x),
            t_high_a_low = bezier_interpolate(dataset, result_size, t_high * t_block_size + 0 * a_block_size, x),
            t_low_a_high = bezier_interpolate(dataset, result_size, t_low * t_block_size + 1 * a_block_size, x),
            t_high_a_high = bezier_interpolate(dataset, result_size, t_high * t_block_size + 1 * a_block_size, x);

        std::vector<ScalarFloat>
            res_a_low = lerp_vectors(t_low_a_low, t_high_a_low, t_rem),
            res_a_high = lerp_vectors(t_low_a_high, t_high_a_high, t_rem);

        std::vector<ScalarFloat> res(result_size);
        for (size_t i = 0; i < result_size; ++i)
            res[i] = dr::lerp(res_a_low[i], res_a_high[i], albedo[i/nb_params]);

        return res;
    }

NAMESPACE_END(mitsuba)
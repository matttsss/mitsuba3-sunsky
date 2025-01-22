#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/quad.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>

#include <mitsuba/render/sunsky/sunsky.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-constant:

Constant environment emitter (:monosp:`constant`)
-------------------------------------------------

.. pluginparameters::
 * - turbidity
   - |float|
   - Atmosphere turbidity (Default: 3, clear sky in a temperate climate).
   - |exposed|, |differentiable|

 * - albedo
   - |float| or |spectrum|
   - Ground albedo (Default: 0.3).
   - |exposed|

 * - sun_direction
   - |vector|
   - Direction of the sun in the sky (No defaults).
   - |exposed|, |differentiable|

 * - sun_scale
   - |float|
   - Scale factor for the sun radiance (Default: 1).
   - |exposed|

 * - sky_scale
   - |float|
   - Scale factor for the sky radiance (Default: 1).
   - |exposed|

This plugin implements an environment emitter for the sun and sky dome.
It is based on the Wilkie-Hosek sun and sky model.

.. tabs::
    .. code-tab:: xml
        :name: constant-light

        <emitter type="constant">
            <rgb name="radiance" value="1.0"/>
        </emitter>

    .. code-tab:: python

        'type': 'constant',
        'radiance': {
            'type': 'rgb',
            'value': 1.0,
        }

 */

#define DATABASE_PATH "resources/sunsky/datasets/"
#define DATABASE_TYPE std::string(is_spectral_v<Spectrum> ? "_spec_" : "_rgb_")

template <typename Float, typename Spectrum>
class SunskyEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_to_world)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    using FloatStorage = DynamicBuffer<Float>;
    using Gaussian = dr::Array<Float, NB_GAUSSIAN_PARAMS>;
    using FullSpectrum = std::conditional_t<is_spectral_v<Spectrum>, mitsuba::Spectrum<Float, NB_WAVELENGTHS>, Spectrum>;

    SunskyEmitter(const Properties &props) : Base(props) {
        if constexpr (!(is_rgb_v<Spectrum> || is_spectral_v<Spectrum>))
            Throw("Unsupported spectrum type!");

        if constexpr (is_spectral_v<Spectrum>)
            c_wavelengths = {320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720};

        init_from_props(props);

        // Extract albedo from texture
        FloatStorage albedo = extract_albedo(m_albedo);

        // ================= UPDATE ANGLES =================
        Vector3f local_sun_dir = m_to_world.value().inverse().transform_affine(m_sun_dir);

        m_sun_angles = from_spherical(local_sun_dir);
        m_local_sun_frame = Frame3f(local_sun_dir);

        const Float sun_eta = 0.5f * dr::Pi<Float> - m_sun_angles.y();

        // ================= GET SKY RADIANCE =================
        m_sky_dataset = array_from_file<Float64, Float>(DATABASE_PATH "sky" + DATABASE_TYPE + "params.bin");
        m_sky_rad_dataset = array_from_file<Float64, Float>(DATABASE_PATH "sky" + DATABASE_TYPE + "rad.bin");

        m_sky_params = compute_radiance_params<SKY_DATASET_SIZE>(m_sky_dataset, albedo, m_turbidity, sun_eta),
        m_sky_radiance = compute_radiance_params<SKY_DATASET_RAD_SIZE>(m_sky_rad_dataset, albedo, m_turbidity, sun_eta);

        // ================= GET SUN RADIANCE =================
        m_sun_rad_dataset = array_from_file<Float64, Float>(DATABASE_PATH "sun" + DATABASE_TYPE + "rad.bin");

        m_sun_radiance = compute_sun_params<SUN_DATASET_SIZE>(m_sun_rad_dataset, m_turbidity);

        // Only used in spectral mode since limb darkening is baked in the RGB dataset
        if constexpr (is_spectral_v<Spectrum>)
            m_sun_ld = array_from_file<Float64, Float>(DATABASE_PATH "sun_spec_ld.bin");


        // ================= GET TGMM TABLES =================
        m_tgmm_tables = array_from_file<Float32, Float>(DATABASE_PATH "tgmm_tables.bin");

        const auto [distrib_params, mis_weights] = compute_tgmm_distribution<TGMM_DATA_SIZE>(m_tgmm_tables, m_turbidity, sun_eta);

        m_gaussians = distrib_params;
        m_gaussian_distr = DiscreteDistribution<Float>(mis_weights);

        m_sky_sampling_w = estimate_sky_sun_ratio();

        std::cout << "Sky sampling weight: " << m_sky_sampling_w << std::endl;

        // ================= GENERAL PARAMETERS =================

        /* Until `set_scene` is called, we have no information
        about the scene and default to the unit bounding sphere. */
        m_bsphere = BoundingSphere3f(ScalarPoint3f(0.f), 1.f);
        m_surface_area = 4.f * dr::Pi<Float>;

        m_flags = +EmitterFlags::Infinite | +EmitterFlags::SpatiallyVarying;
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        using SpecUInt32 = dr::uint32_array_t<Spectrum>;
        using SpecMask   = dr::mask_t<Spectrum>;

        Vector3f local_wo = m_to_world.value().inverse().transform_affine(-si.wi);
        Float cos_theta = Frame3f::cos_theta(local_wo),
              gamma = dr::unit_angle(Vector3f(m_local_sun_frame.n), local_wo);

        active &= cos_theta >= 0;
        Mask hit_sun = active & (dr::dot(m_local_sun_frame.n, local_wo) >= dr::cos(m_sun_half_aperture));

        UnpolarizedSpectrum res = 0.f;
        if constexpr (is_rgb_v<Spectrum>) {
            // dr::width(idx) == 1
            const SpecUInt32 idx = SpecUInt32({0, 1, 2});

            res = m_sky_scale * render_sky<Spectrum>(idx, cos_theta, gamma, active);

            // FIXME: explain this constant
            res += m_sun_scale * render_sun<Spectrum>(idx, cos_theta, gamma, hit_sun) * 467.069280386 * get_area_ratio(m_sun_half_aperture);

            res *= MI_CIE_Y_NORMALIZATION;

        } else {
            const Spectrum normalized_wavelengths = (si.wavelengths - WAVELENGTHS<ScalarFloat>[0]) / WAVELENGTH_STEP;
            const SpecMask valid_idx = (0.f <= normalized_wavelengths) & (normalized_wavelengths <= NB_CHANNELS - 1);

            const SpecUInt32 query_idx_low  = dr::floor2int<SpecUInt32>(normalized_wavelengths),
                             query_idx_high = query_idx_low + 1;

            const Spectrum lerp_factor = normalized_wavelengths - query_idx_low;

            res = m_sky_scale * dr::lerp(
                render_sky<Spectrum>(query_idx_low, cos_theta, gamma, active & valid_idx),
                render_sky<Spectrum>(query_idx_high, cos_theta, gamma, active & valid_idx),
                lerp_factor);

            Spectrum sun_rad_low  = render_sun<Spectrum>(query_idx_low, cos_theta, gamma, hit_sun & valid_idx),
                     sun_rad_high = render_sun<Spectrum>(query_idx_high, cos_theta, gamma, hit_sun & valid_idx),
                     sun_rad = dr::lerp(sun_rad_low, sun_rad_high, lerp_factor);

            Spectrum sun_ld = compute_sun_ld<Spectrum>(query_idx_low, query_idx_high, lerp_factor, gamma, hit_sun & valid_idx);

            res += m_sun_scale * sun_rad * sun_ld * get_area_ratio(m_sun_half_aperture);

        }

        return res & active;
    }


    std::pair<DirectionSample3f, Spectrum> sample_direction(const Interaction3f &it,
                                                            const Point2f &sample,
                                                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        const Mask pick_sky = sample.x() < m_sky_sampling_w;

        // Sample the sun or the sky
        Vector3f sample_dir = dr::select(
                pick_sky,
                sample_sky({sample.x() / m_sky_sampling_w, sample.y()}, active),
                sample_sun({(sample.x() - m_sky_sampling_w) / (1 - m_sky_sampling_w), sample.y()})
        );

        //sample_dir = m_local_sun_frame.n;
        active &= Frame3f::cos_theta(sample_dir) >= 0.f;

        // Automatically enlarge the bounding sphere when it does not contain the reference point
        Float radius = dr::maximum(m_bsphere.radius, dr::norm(it.p - m_bsphere.center)),
              dist   = 2.f * radius;

        Vector3f d = m_to_world.value().transform_affine(sample_dir);
        DirectionSample3f ds = dr::zeros<DirectionSample3f>();
        ds.p       = dr::fmadd(d, dist, it.p);
        ds.n       = -d;
        ds.uv      = sample;
        ds.time    = it.time;
        ds.delta   = false;
        ds.emitter = this;
        ds.d       = d;
        ds.dist    = dist;

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.wi = -d;
        si.wavelengths = it.wavelengths;

        auto [sky_pdf, sun_pdf] = compute_pdfs(sample_dir, !pick_sky, active);

        ds.pdf = dr::lerp(sun_pdf, sky_pdf, m_sky_sampling_w);

        Spectrum res = eval(si, active) / ds.pdf;
                 res = dr::select(dr::isfinite(res), res, 0.f);
        return { ds, res };
    }

    Float pdf_direction(const Interaction3f &, const DirectionSample3f &ds, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        Vector3f local_dir = m_to_world.value().inverse().transform_affine(ds.d);
        const auto [sky_pdf, sun_pdf] = compute_pdfs(local_dir, true, active);

        Float combined_pdf = dr::lerp(sun_pdf, sky_pdf, m_sky_sampling_w);
        return dr::select(active, combined_pdf, 0.f);
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("turbidity", m_turbidity, +ParamFlags::Differentiable);
        callback->put_parameter("sky_scale", m_sky_scale, +ParamFlags::NonDifferentiable);
        callback->put_parameter("sun_scale", m_sun_scale, +ParamFlags::NonDifferentiable);
        callback->put_object("albedo", m_albedo.get(), +ParamFlags::NonDifferentiable);
        if (m_active_record) {
            callback->put_parameter("hour", m_time.hour, +ParamFlags::Differentiable);
        } else {
            callback->put_parameter("sun_direction", m_sun_dir, +ParamFlags::Differentiable);
        }
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        const bool turbidity_changed = string::contains(keys, "turbidity");
        const bool albedo_changed    = string::contains(keys, "albedo");
        const bool sun_dir_changed   = string::contains(keys, "sun_direction") || string::contains(keys, "hour");

        // Reassigns array and "destroys" the gradient flow
        // m_turbidity = dr::clip(m_turbidity, 1.f, 10.f);
        dr::set_label(m_turbidity, "turbidity");

        FloatStorage albedo = extract_albedo(m_albedo);

        // Update sun angles
        if (sun_dir_changed) {
            Vector3f local_sun_dir;
            if (m_active_record) {
                local_sun_dir = compute_sun_coordinates(m_time, m_location);
                m_sun_dir = m_to_world.value().transform_affine(local_sun_dir);
            } else {
                local_sun_dir = m_to_world.value().inverse().transform_affine(m_sun_dir);
            }
            m_sun_angles = from_spherical(local_sun_dir);
            m_local_sun_frame = Frame3f(local_sun_dir);
        }

        Float eta = 0.5f * dr::Pi<Float> - m_sun_angles.y();

        // Update sky
        if (turbidity_changed || albedo_changed || sun_dir_changed) {
            m_sky_params = compute_radiance_params<SKY_DATASET_SIZE>(m_sky_dataset, albedo, m_turbidity, eta);
            m_sky_radiance = compute_radiance_params<SKY_DATASET_RAD_SIZE>(m_sky_rad_dataset, albedo, m_turbidity, eta);
        }

        // Update sun
        if (turbidity_changed)
            m_sun_radiance = compute_sun_params<SUN_DATASET_SIZE>(m_sun_rad_dataset, m_turbidity);


        // Update TGMM
        if (turbidity_changed || sun_dir_changed) {
            const auto [gaussian_params, mis_weights] = compute_tgmm_distribution<TGMM_DATA_SIZE>(m_tgmm_tables, m_turbidity, eta);
            m_gaussians = gaussian_params;
            m_gaussian_distr = DiscreteDistribution<Float>(mis_weights);
        }

        m_sky_sampling_w = estimate_sky_sun_ratio();

        dr::set_label(m_sky_params, "sky_params");
        dr::set_label(m_sky_radiance, "sky_radiance");
        dr::set_label(m_sun_radiance, "sun_radiance");
        dr::set_label(m_gaussians, "gaussian_params");
        dr::set_label(m_sky_sampling_w, "sky sampling weight");
    }

    void set_scene(const Scene *scene) override {
        if (scene->bbox().valid()) {
            ScalarBoundingSphere3f scene_sphere =
                scene->bbox().bounding_sphere();
            m_bsphere = BoundingSphere3f(scene_sphere.center, scene_sphere.radius);
            m_bsphere.radius =
                dr::maximum(math::RayEpsilon<Float>,
                        m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
        } else {
            m_bsphere.center = 0.f;
            m_bsphere.radius = math::RayEpsilon<Float>;
        }
        m_surface_area = 4.f * dr::Pi<ScalarFloat> * dr::square(m_bsphere.radius);

        dr::make_opaque(m_bsphere.center, m_bsphere.radius, m_surface_area);
    }


    /// This emitter does not occupy any particular region of space, return an invalid bounding box
    ScalarBoundingBox3f bbox() const override {
        return ScalarBoundingBox3f();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SunskyEmitter[" << std::endl
            << "  bsphere = " << string::indent(m_bsphere) << std::endl
            << "  turbidity = " << string::indent(m_turbidity) << std::endl
            << "  sky_scale = " << string::indent(m_sky_scale) << std::endl
            << "  sun_scale = " << string::indent(m_sun_scale) << std::endl
            << "  albedo = " << string::indent(m_albedo) << std::endl
            << "  sun aperture (°) = " << string::indent(dr::rad_to_deg(2 * m_sun_half_aperture)) << std::endl;
        if (m_active_record) {
            oss << "  location = " << m_location.to_string() << std::endl
                << "  date_time = " << m_time.to_string() << std::endl;
        } else {
            oss << "  sun_dir = " << string::indent(m_sun_dir) << std::endl;
        }
        oss << "]";
        return oss.str();
    }


    MI_DECLARE_CLASS()
private:

    /**
     * Renders the sky for the given channel indices and angles
     *
     * Based on the Hosek-Wilkie skylight model
     * https://cgg.mff.cuni.cz/projects/SkylightModelling/HosekWilkie_SkylightModel_SIGGRAPH2012_Preprint_lowres.pdf
     *
     * @tparam Spec Spectral type to render (adapts the number of channels)
     * @param channel_idx Indices of the channels to render
     * @param cos_theta Cosine of the angle between the z-axis (up) and the viewing direction
     * @param gamma Angle between the sun and the viewing direction
     * @param active Mask for the active lanes and channel idx
     * @return The rendered sky radiance
     */
    template <typename Spec> Spec render_sky(
        const dr::uint32_array_t<Spec>& channel_idx,
        const Float& cos_theta, const Float& gamma,
        const dr::mask_t<Spec>& active) const {

        // Gather coefficients for the skylight equation
        using SpecSkyParams = dr::Array<Spec, NB_SKY_PARAMS>;
        SpecSkyParams coefs = dr::gather<SpecSkyParams>(m_sky_params, channel_idx, active);

        Float cos_gamma = dr::cos(gamma),
              cos_gamma_sqr = dr::square(cos_gamma);

        Spec c1 = 1 + coefs[0] * dr::exp(coefs[1] / (cos_theta + 0.01f));
        Spec chi = (1 + cos_gamma_sqr) / dr::pow(1 + dr::square(coefs[8]) - 2 * coefs[8] * cos_gamma, 1.5);
        Spec c2 = coefs[2] + coefs[3] * dr::exp(coefs[4] * gamma) + coefs[5] * cos_gamma_sqr + coefs[6] * chi + coefs[7] * dr::safe_sqrt(cos_theta);

        return c1 * c2 * dr::gather<Spec>(m_sky_radiance, channel_idx, active);
    }

   /**
    * Renders the sun for the given channel indices and angles
    * The template parameter is used to render the full 11 wavelengths at once
    * in pre-computations
    *
    * Based on the Hosek-Wilkie sun model
    * https://cgg.mff.cuni.cz/publications/adding-a-solar-radiance-function-to-the-hosek-wilkie-skylight-model/
    *
    * @tparam Spec Spectral type to render (adapts the number of channels)
    * @param channel_idx Indices of the channels to render
    * @param cos_theta Cosine of the angle between the z-axis (up) and the viewing direction
    * @param gamma Angle between the sun and the viewing direction
    * @param active Mask for the active lanes and channel idx
    * @return The rendered sun radiance
    */
    template <typename Spec> Spec render_sun(
        const dr::uint32_array_t<Spec>& channel_idx,
        const Float& cos_theta, const Float& gamma,
        const dr::mask_t<Spec>& active) const {

        using SpecUInt32 = dr::uint32_array_t<Spec>;

        // Angles computation
        Float elevation =  0.5f * dr::Pi<Float> - dr::acos(cos_theta);

        // Find the segment of the piecewise function we are in
        UInt32 pos = dr::floor2int<UInt32>(dr::pow(2 * elevation * dr::InvPi<Float>, 1.f/3.f) * NB_SUN_SEGMENTS);
               pos = dr::minimum(pos, NB_SUN_SEGMENTS - 1);

        const Float break_x = 0.5f * dr::Pi<Float> * dr::pow((Float)pos / NB_SUN_SEGMENTS, 3.f),
                    x = elevation - break_x;

        Spec solar_radiance = 0.f;
        if constexpr (is_spectral_v<Spec>) {
            DRJIT_MARK_USED(gamma);
            // Compute sun radiance
            SpecUInt32 global_idx = pos * NB_WAVELENGTHS * NB_SUN_CTRL_PTS + channel_idx * NB_SUN_CTRL_PTS;
            for (uint8_t k = 0; k < NB_SUN_CTRL_PTS; ++k)
                solar_radiance += dr::pow(x, k) * dr::gather<Spec>(m_sun_radiance, global_idx + k, active);

        } else {
            // Reproduces the spectral equation above but distributes the product of sums
            // since it uses interpolated coefficients from the spectral dataset

            // Angles computation
            //Float64 sin_gamma_sqr      = dr::fnmadd(cos_gamma, cos_gamma, 1.f),
            //        sin_sun_cutoff_sqr = dr::fnmadd(SUN_COS_CUTOFF, SUN_COS_CUTOFF, 1.f);
            //Float64 cos_psi = dr::safe_sqrt(1 - sin_gamma_sqr / sin_sun_cutoff_sqr);

            const ScalarFloat64 sol_rad_sin = dr::sin(m_sun_half_aperture);
            const ScalarFloat64 ar2 = 1 / ( sol_rad_sin * sol_rad_sin );
            const Float64 singamma = (Float64) dr::sin(gamma);
            const Float64 sc2 = 1.0 - ar2 * singamma * singamma;
            const Float cos_psi = dr::safe_sqrt(sc2);

            SpecUInt32 global_idx = pos * (3 * NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS) +
                                    channel_idx * (NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS);

#if 0
            UInt32 k_ = 0, j_ = 0;
            std::tie(k_, j_, solar_radiance) = dr::while_loop(
                std::make_tuple(k_, j_, solar_radiance),
                [](const UInt32& k, const UInt32&, const Spec&) {
                    return k < NB_SUN_CTRL_PTS;
                },
                [&](UInt32& k, UInt32& j, Spec& solar_radiance_) {
                    solar_radiance_ += dr::pow(x, k) * dr::pow(cos_psi, j) *
                                      dr::gather<Spec>(m_sun_radiance, global_idx + k * NB_SUN_LD_PARAMS + j, active);

                    j += 1;
                    k = dr::select(j == NB_SUN_LD_PARAMS, k + 1, k);
                    j = dr::select(j == NB_SUN_LD_PARAMS, 0, j);
                },
                "Sun radiance computation"
            );
#elif 0
            using SpecArray = dr::Array<Spec, NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS>;
            using UInt32Array = dr::Array<UInt32, NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS>;

            UInt32Array idx = dr::arange<UInt32Array>(NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS),
                        k_idx = idx / NB_SUN_LD_PARAMS,
                        j_idx = idx % NB_SUN_LD_PARAMS;

            SpecArray pow_x = dr::pow(x, k_idx),
                      pow_cos_psi = dr::pow(cos_psi, j_idx),
                      sun_rad = dr::gather<SpecArray>(m_sun_radiance, global_idx, active);

            solar_radiance = dr::sum(pow_x * pow_cos_psi * sun_rad);
#else
            for (uint8_t k = 0; k < NB_SUN_CTRL_PTS; ++k) {
                for (uint8_t j = 0; j < NB_SUN_LD_PARAMS; ++j) {
                    solar_radiance += dr::pow(x, k) * dr::pow(cos_psi, j) *
                                      dr::gather<Spec>(m_sun_radiance, global_idx + k * NB_SUN_LD_PARAMS + j, active);
                }
            }
#endif
        }

        return solar_radiance & active;

    }

    /**
     * \brief Computes the limb darkening of the sun for a given gamma.
     * Only works for spectral mode since limb darkening is backed into the RGB
     * model
     *
     * @tparam Spec Spectral type to render (adapts the number of channels)
     * @param channel_idx_low Indices of the lower wavelengths
     * @param channel_idx_high Indices of the upper wavelengths
     * @param lerp_f Linear interpolation factor for wavelength
     * @param gamma Angle between the sun's center and the viewing ray
     * @param active Indicates if the channel indices are valid and that the sun
     * was hit
     * @return The spectral values of limb darkening to apply to the sun's
     * radiance by multiplication
     */
    template <typename Spec> Spec compute_sun_ld(
            const dr::uint32_array_t<Spec>& channel_idx_low,
            const dr::uint32_array_t<Spec>& channel_idx_high,
            const Spec& lerp_f, const Float& gamma,
            const dr::mask_t<Spec>& active) const {

        using SpecLdArray = dr::Array<Spec, NB_SUN_LD_PARAMS>;

        SpecLdArray sun_ld_low  = dr::gather<SpecLdArray>(m_sun_ld, channel_idx_low, active),
                    sun_ld_high = dr::gather<SpecLdArray>(m_sun_ld, channel_idx_high, active),
                    sun_ld_coefs = dr::lerp(sun_ld_low, sun_ld_high, lerp_f);

        // Angles computation
        //Float64 sin_gamma_sqr      = dr::fnmadd(cos_gamma, cos_gamma, 1.f),
        //        sin_sun_cutoff_sqr = dr::fnmadd(SUN_COS_CUTOFF, SUN_COS_CUTOFF, 1.f);
        //Float64 cos_psi = dr::safe_sqrt(1 - sin_gamma_sqr / sin_sun_cutoff_sqr);

        const ScalarFloat64 sol_rad_sin = dr::sin(m_sun_half_aperture);
        const ScalarFloat64 ar2 = 1 / ( sol_rad_sin * sol_rad_sin );
        const Float64 singamma = dr::sin(gamma);
        const Float64 sc2 = 1.0 - ar2 * singamma * singamma;
        const Float cos_psi = dr::safe_sqrt(sc2);

        Spec sun_ld = 0.f;
        for (uint8_t j = 0; j < NB_SUN_LD_PARAMS; ++j)
            sun_ld += dr::pow(cos_psi, j) * sun_ld_coefs[j];

        return sun_ld & active;
    }

    MI_INLINE Point2f gaussian_cdf(const Point2f& mu, const Point2f& sigma, const Point2f& x) const {
        return 0.5f * (1 + dr::erf(dr::InvSqrtTwo<Float> * (x - mu) / sigma));
    }

    /**
     * Samples the sky from the truncated gaussian mixture with the given sample
     * Based on the Truncated Gaussian Mixture Model (TGMM) for sky dome by N. Vitsas and K. Vardis
     * https://diglib.eg.org/items/b3f1efca-1d13-44d0-ad60-741c4abe3d21
     *
     * @param sample Sample uniformly distributed in [0, 1]^2
     * @param active Mask for the active lanes
     * @return The sampled direction in the sky and its PDF
     */
    Vector3f sample_sky(Point2f sample, const Mask& active) const {
        // Sample a gaussian from the mixture
        const auto [idx, temp_sample] = m_gaussian_distr.sample_reuse(sample.x(), active);

        // {mu_phi, mu_theta, sigma_phi, sigma_theta, weight}
        Gaussian gaussian = dr::gather<Gaussian>(m_gaussians, idx, active);

        const Point2f a = { 0.0 },
                      b = { dr::TwoPi<Float>, 0.5f * dr::Pi<Float> };

        const Point2f mu    = { gaussian[0], gaussian[1] },
                      sigma = { gaussian[2], gaussian[3] };

        const Point2f cdf_a = gaussian_cdf(mu, sigma, a),
                      cdf_b = gaussian_cdf(mu, sigma, b);

        sample = dr::lerp(cdf_a, cdf_b, Point2f{temp_sample, sample.y()});
        // Clamp to erfinv's domain of definition
        sample = dr::clip(sample, dr::Epsilon<Float>, dr::OneMinusEpsilon<Float>);

        Point2f angles = dr::SqrtTwo<Float> * dr::erfinv(2 * sample - 1) * sigma + mu;

        // From fixed reference frame where sun_phi = pi/2 to local frame
        angles.x() += m_sun_angles.x() - 0.5f * dr::Pi<Float>;
        // Clamp theta to avoid negative z-axis values (FP errors)
        angles.y() = dr::minimum(angles.y(), 0.5f * dr::Pi<Float> - dr::Epsilon<Float>);

        return m_to_world.value().transform_affine(to_spherical(angles));
    }

    Vector3f sample_sun(const Point2f& sample) const {
        return m_local_sun_frame.to_world(
            warp::square_to_uniform_cone(sample, (Float) dr::cos(m_sun_half_aperture))
        );
    }

    /**
     * Computes the PDFs of the sky and sun for the given local direction
     *
     * @param local_dir Local direction in the sky
     * @param check_sun Indicates if the sun's intersection should be tested
     * @param active Mask for the active lanes
     * @return The sky and sun PDFs
     */
    std::pair<Float, Float> compute_pdfs(const Vector3f& local_dir, const Mask& check_sun, Mask active) const {
        // Check for bounds on PDF
        Float sin_theta = Frame3f::sin_theta(local_dir);
        active &= (Frame3f::cos_theta(local_dir) >= 0.f) && (sin_theta != 0.f);
        sin_theta = dr::maximum(sin_theta, SIN_OFFSET);

        const Float cosine_cutoff = dr::cos(m_sun_half_aperture);
        Float sky_pdf = tgmm_pdf(from_spherical(local_dir), active) / sin_theta,
              sun_pdf = warp::square_to_uniform_cone_pdf(m_local_sun_frame.to_local(local_dir), cosine_cutoff);
              sun_pdf = dr::select(!check_sun || dr::dot(m_local_sun_frame.n, local_dir) >= cosine_cutoff, sun_pdf, 0.f);

        return {sky_pdf, sun_pdf};
    }

    Float tgmm_pdf(Point2f angles, Mask active) const {
        // From local frame to reference frame where sun_phi = pi/2 and phi is in [0, 2pi]
        angles.x() -= m_sun_angles.x() - 0.5f * dr::Pi<Float>;
        angles.x() = dr::select(angles.x() < 0, angles.x() + dr::TwoPi<Float>, angles.x());
        angles.x() = dr::select(angles.x() > dr::TwoPi<Float>, angles.x() - dr::TwoPi<Float>, angles.x());

        // Bounds check for theta
        active &= (angles.y() >= 0.f) && (angles.y() <= 0.5f * dr::Pi<Float>);

        // Bounding points of the truncated gaussian mixture
        const Point2f a = { 0.0 },
                      b = { dr::TwoPi<Float>, 0.5f * dr::Pi<Float> };

        Float pdf = 0.0;
        for (size_t i = 0; i < 4 * NB_GAUSSIAN; ++i) {
            const size_t base_idx = i * NB_GAUSSIAN_PARAMS;
            // {mu_phi, mu_theta}, {sigma_phi, sigma_theta}
            Point2f mu    = { m_gaussians[base_idx + 0], m_gaussians[base_idx + 1] },
                    sigma = { m_gaussians[base_idx + 2], m_gaussians[base_idx + 3] };

            Point2f cdf_a = gaussian_cdf(mu, sigma, a),
                    cdf_b = gaussian_cdf(mu, sigma, b);

            Float volume = (cdf_b.x() - cdf_a.x()) * (cdf_b.y() - cdf_a.y()) * (sigma.x() * sigma.y());

            Point2f sample = (angles - mu) / sigma;
            Float gaussian_pdf = warp::square_to_std_normal_pdf(sample);

            pdf += m_gaussians[base_idx + 4] * gaussian_pdf / volume;
        }
        return dr::select(active, pdf, 0.f);
    }

    FloatStorage extract_albedo(const ref<Texture>& albedo_tex) const {
        FloatStorage albedo = dr::zeros<FloatStorage>(NB_CHANNELS);
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

        if constexpr (is_rgb_v<Spectrum>) {
            albedo = dr::ravel(albedo_tex->eval(si));

        } else if constexpr (dr::is_array_v<Float> && is_spectral_v<Spectrum>) {
            si.wavelengths = dr::load<FloatStorage>(WAVELENGTHS<ScalarFloat>, NB_CHANNELS);
            albedo = albedo_tex->eval(si)[0];

        } else if (!dr::is_array_v<Float> && is_spectral_v<Spectrum>) {
            for (ScalarUInt32 i = 0; i < NB_CHANNELS; ++i) {
                si.wavelengths = WAVELENGTHS<ScalarFloat>[i];
                dr::scatter(albedo, albedo_tex->eval(si)[0], (UInt32) i);
            }
        }

        albedo = dr::clip(albedo, 0.f, 1.f);

        return albedo;
    }

    /**
     * \brief Estimates the ratio of sky to sun luminance over the hemisphere,
     * can be used to estimate the sampling weight of the sun and sky.
     *
     * @return The sky's ratio of luminance, in [0, 1]
     */
    Float estimate_sky_sun_ratio() const {
        FullSpectrum sky_radiance = dr::zeros<FullSpectrum>(),
                     sun_radiance = dr::zeros<FullSpectrum>();

        dr::uint32_array_t<FullSpectrum> channel_idx;
        if constexpr (is_rgb_v<Spectrum>)
            channel_idx = {0, 1, 2};
        else
            channel_idx = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        if constexpr (!dr::is_array_v<Float>)
            return 0.5f;
        else {

            // Quadrature points and weights
            const auto [x, w_x] = quad::gauss_legendre<Float>(200);

            // Compute sky radiance over hemisphere
            {

                // Mapping for [-1, 1] x [-1, 1] -> [0, 2pi] x [0, 1]
                const Float inv_J = 0.5f * dr::Pi<Float>;
                Float phi = dr::Pi<Float> * (x + 1),
                      cos_theta = 0.5f * (x + 1);

                std::tie(phi, cos_theta) = dr::meshgrid(phi, cos_theta);
                const auto [w_phi, w_cos_theta] = dr::meshgrid(w_x, w_x);

                const auto [sin_phi, cos_phi] = dr::sincos(phi);
                const auto sin_theta = dr::safe_sqrt(1 - dr::square(cos_theta));

                Vector3f sky_wo {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
                Float gamma = dr::unit_angle(Vector3f(m_local_sun_frame.n), sky_wo);

                FullSpectrum ray_radiance = render_sky<FullSpectrum>(channel_idx, cos_theta, gamma, true) * w_phi * w_cos_theta;
                sky_radiance = dr::sum_inner(ray_radiance) * inv_J;
            }

            // Compute sun radiance over hemisphere
            {
                const ScalarFloat cosine_cutoff = dr::cos(m_sun_half_aperture);

                // Mapping for [-1, 1] x [-1, 1] -> [0, 2pi] x [cos(alpha/2), 1]
                const Float inv_J = 0.5f * dr::Pi<ScalarFloat> * (1 - cosine_cutoff);
                Float phi = dr::Pi<Float> * (x + 1),
                      cos_gamma = 0.5f * ((1 - cosine_cutoff) * x + (1 + cosine_cutoff));

                std::tie(phi, cos_gamma) = dr::meshgrid(phi, cos_gamma);
                const auto [w_phi, w_cos_theta] = dr::meshgrid(w_x, w_x);

                const auto sin_gamma = dr::safe_sqrt(1 - dr::square(cos_gamma));
                const auto [sin_phi, cos_phi] = dr::sincos(phi);

                // View ray in local sun coordinates
                Vector3f sun_wo {sin_gamma * cos_phi, sin_gamma * sin_phi, cos_gamma};
                Float gamma = dr::unit_angle_z(sun_wo);

                // View ray in local emitter coordinates
                sun_wo = m_local_sun_frame.to_world(sun_wo);

                Float cos_theta = Frame3f::cos_theta(sun_wo);

                Mask active = cos_theta >= 0;
                FullSpectrum ray_radiance = render_sun<FullSpectrum>(channel_idx, cos_theta, gamma, active) * w_phi * w_cos_theta;

                // Apply sun limb darkening if not already
                if constexpr (is_spectral_v<Spectrum>)
                    ray_radiance *= compute_sun_ld<FullSpectrum>(channel_idx, channel_idx, 0.f, gamma, active);

                sun_radiance = dr::sum_inner(ray_radiance) * inv_J;
            }

            // Extract luminance
            Float sky_lum = m_sky_scale, sun_lum = m_sun_scale;
            if constexpr (is_rgb_v<Spectrum>) {
                sky_lum *= luminance(sky_radiance);
                sun_lum *= luminance(sun_radiance) * 467.069280386 * get_area_ratio(m_sun_half_aperture);
            } else {
                sky_lum *= luminance(sky_radiance, c_wavelengths);
                sun_lum *= luminance(sun_radiance, c_wavelengths) * get_area_ratio(m_sun_half_aperture);
            }

            // Normalize quantities for valid distribution
            return sky_lum / (sky_lum + sun_lum);
        }
    }


    void init_from_props(const Properties& props) {
        m_sun_scale = props.get<ScalarFloat>("sun_scale", 1.f);
        m_sky_scale = props.get<ScalarFloat>("sky_scale", 1.f);

        m_turbidity = props.get<ScalarFloat>("turbidity", 3.f);
        dr::make_opaque(m_turbidity);

        m_sun_half_aperture = dr::deg_to_rad(0.5f * props.get<ScalarFloat>("sun_aperture", 0.5358));
        if (m_sun_half_aperture <= 0.f || 0.5f * dr::Pi<Float> <= m_sun_half_aperture)
            Throw("Invalid sun aperture angle, must be in ]0, 90[ degrees!");

        m_albedo = props.texture<Texture>("albedo", 1.f);
        if (m_albedo->is_spatially_varying())
            Throw("Expected a non-spatially varying radiance spectra!");

        if (props.has_property("sun_direction")) {
            if (props.has_property("latitude") || props.has_property("longitude")
                || props.has_property("timezone") || props.has_property("day")
                || props.has_property("time"))
                Log(Error, "Both the 'sun_direction' parameter and time/location "
                           "information were provided -- only one of them can be specified at a time!");
            m_active_record = false;
            m_sun_dir = dr::normalize(props.get<ScalarVector3f>("sun_direction"));
            dr::make_opaque(m_sun_dir);

        } else {
            m_location.latitude  = props.get<ScalarFloat>("latitude", 35.6894f);
            m_location.longitude = props.get<ScalarFloat>("longitude", 139.6917f);
            m_location.timezone  = props.get<ScalarFloat>("timezone", 9);
            m_time.year          = props.get<ScalarInt32>("year", 2010);
            m_time.day           = props.get<ScalarUInt32>("day", 10);
            m_time.month         = props.get<ScalarUInt32>("month", 7);
            m_time.hour          = props.get<ScalarFloat>("hour", 15.0f);
            m_time.minute        = props.get<ScalarFloat>("minute", 0.0f);
            m_time.second        = props.get<ScalarFloat>("second", 0.0f);

            m_active_record = true;
            dr::make_opaque(m_location.latitude, m_location.longitude, m_location.timezone,
                            m_time.year, m_time.day, m_time.month, m_time.hour, m_time.minute, m_time.second);

            m_sun_dir = compute_sun_coordinates(m_time, m_location);
            m_sun_dir = m_to_world.value().transform_affine(m_sun_dir);
        }

    }

    // ================================================================================================
    // ========================================= ATTRIBUTES ===========================================
    // ================================================================================================

    /// Offset used to avoid division by zero in the pdf computation
    static constexpr ScalarFloat SIN_OFFSET = dr::Epsilon<Float>; // chi2 passes with 0.00775
    /// Number of channels used in the skylight model
    static constexpr uint32_t NB_CHANNELS = is_spectral_v<Spectrum> ? NB_WAVELENGTHS : 3;

    // Dataset sizes
    static constexpr uint32_t SKY_DATASET_SIZE = NB_TURBIDITY * NB_ALBEDO * NB_SKY_CTRL_PTS * NB_CHANNELS * NB_SKY_PARAMS,
                              SKY_DATASET_RAD_SIZE = NB_TURBIDITY * NB_ALBEDO * NB_SKY_CTRL_PTS * NB_CHANNELS,
                              SUN_DATASET_SIZE = NB_TURBIDITY * NB_CHANNELS * NB_SUN_SEGMENTS * NB_SUN_CTRL_PTS * (is_spectral_v<Spectrum> ? 1 : NB_SUN_LD_PARAMS),
                              TGMM_DATA_SIZE = (NB_TURBIDITY - 1) * NB_ETAS * NB_GAUSSIAN * NB_GAUSSIAN_PARAMS;

    FullSpectrum c_wavelengths;

    Float m_surface_area;
    BoundingSphere3f m_bsphere;

    Float m_turbidity;
    Float m_sky_sampling_w;
    ScalarFloat m_sky_scale;
    ScalarFloat m_sun_scale;

    ref<Texture> m_albedo;

    // Sun parameters
    Vector3f m_sun_dir;
    Point2f m_sun_angles;
    Frame3f m_local_sun_frame;
    ScalarFloat m_sun_half_aperture;
    bool m_active_record;
    LocationRecord<Float> m_location;
    DateTimeRecord<Float> m_time;

    // Radiance parameters
    FloatStorage m_sky_params;
    FloatStorage m_sky_radiance;
    FloatStorage m_sun_radiance;

    // Sampling parameters
    FloatStorage m_gaussians;
    DiscreteDistribution<Float> m_gaussian_distr;

    // Permanent datasets loaded from files/memory
    FloatStorage m_sky_dataset;
    FloatStorage m_sky_rad_dataset;
    FloatStorage m_sun_ld; // Not initialized in RGB mode
    FloatStorage m_sun_rad_dataset;
    FloatStorage m_tgmm_tables;

};

MI_IMPLEMENT_CLASS_VARIANT(SunskyEmitter, Emitter)
MI_EXPORT_PLUGIN(SunskyEmitter, "Sun and Sky dome background emitter")
NAMESPACE_END(mitsuba)

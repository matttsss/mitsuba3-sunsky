#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>

#include <mitsuba/render/sunsky/sunsky.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-constant:

Constant environment emitter (:monosp:`constant`)
-------------------------------------------------

.. pluginparameters::

 * - radiance
   - |spectrum|
   - Specifies the emitted radiance in units of power per unit area per unit steradian.
   - |exposed|, |differentiable|

This plugin implements a constant environment emitter, which surrounds
the scene and radiates diffuse illumination towards it. This is often
a good default light source when the goal is to visualize some loaded
geometry that uses basic (e.g. diffuse) materials.

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


#define DATABASE_PREFIX "ssm_dataset"
#define DATABASE_PATH "sunsky-testing/res/datasets/"

template <typename Float, typename Spectrum>
class SunskyEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_to_world)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    using FloatStorage = DynamicBuffer<Float>;

    using Gaussian = dr::Array<Float, NB_GAUSSIAN_PARAMS>;

    using SpecUInt32 = dr::uint32_array_t<Spectrum>;
    using SpecMask = dr::mask_t<Spectrum>;

    SunskyEmitter(const Properties &props) : Base(props) {
        m_sun_scale = props.get<ScalarFloat>("sun_scale", 1.f);
        m_sky_scale = props.get<ScalarFloat>("sky_scale", 1.f);

        m_turbidity = props.get<ScalarFloat>("turbidity", 3.f);

        m_albedo = props.texture<Texture>("albedo", 1.f);
        if (m_albedo->is_spatially_varying())
            Throw("Expected a non-spatially varying radiance spectra!");

        dr::make_opaque(m_turbidity, m_albedo);

        // ================= GET ALBEDO =================
        dr::eval(m_albedo);
        FloatStorage albedo = extract_albedo(m_albedo);

        // ================= GET ANGLES =================
        Vector3f local_sun_dir = dr::normalize(
            m_to_world.scalar().inverse()
            .transform_affine(props.get<ScalarVector3f>("sun_direction")));

        Float sun_theta = dr::unit_angle_z(local_sun_dir),
              sun_eta = 0.5f * dr::Pi<Float> - sun_theta;

        m_sun_phi = dr::atan2(local_sun_dir.y(), local_sun_dir.x());
        m_sun_phi = dr::select(m_sun_phi >= 0, m_sun_phi, m_sun_phi + dr::TwoPi<Float>);

        m_local_sun_frame = Frame3f(local_sun_dir);

        dr::make_opaque(m_sun_phi, m_local_sun_frame);

        // ================= GET SKY RADIANCE =================
        m_sky_dataset = array_from_file<Float64, Float>(DATABASE_PATH + DATASET_NAME + ".bin");
        m_sky_rad_dataset = array_from_file<Float64, Float>(DATABASE_PATH + DATASET_NAME + "_rad.bin");

        m_sky_params = compute_radiance_params<SKY_DATASET_SIZE>(m_sky_dataset, albedo, m_turbidity, sun_eta),
        m_sky_radiance = compute_radiance_params<SKY_DATASET_RAD_SIZE>(m_sky_rad_dataset, albedo, m_turbidity, sun_eta);

        dr::make_opaque(m_sky_params, m_sky_radiance);

        // ================= GET SUN RADIANCE =================
        m_sun_rad_dataset = array_from_file<Float64, Float>(DATABASE_PATH + DATASET_NAME + "_solar.bin");

        m_sun_radiance = compute_sun_params<SUN_DATASET_SIZE>(m_sun_rad_dataset, m_turbidity);

        dr::make_opaque(m_sun_radiance);

        // Only used in spectral mode since limb darkening is baked in the RGB dataset
        if constexpr (is_spectral_v<Spectrum>)
            m_sun_ld = array_from_file<Float64, Float>(DATABASE_PATH DATABASE_PREFIX "_ld_sun.bin");


        // ================= GET TGMM TABLES =================
        m_tgmm_tables = array_from_file<Float, Float>(DATABASE_PATH "tgmm_tables.bin");

        const auto [distrib_params, mis_weights] = compute_tgmm_distribution<TGMM_DATA_SIZE>(m_tgmm_tables, m_turbidity, sun_eta);

        m_gaussians = distrib_params;
        m_gaussian_distr = DiscreteDistribution<Float>(mis_weights);

        dr::make_opaque(m_gaussians, m_gaussian_distr);

        // ================= GENERAL PARAMETERS =================

        /* Until `set_scene` is called, we have no information
        about the scene and default to the unit bounding sphere. */
        m_bsphere = BoundingSphere3f(ScalarPoint3f(0.f), 1.f);
        m_surface_area = 4.f * dr::Pi<Float>;

        m_d65 = Texture::D65(1.f);

        m_flags = +EmitterFlags::Infinite | +EmitterFlags::SpatiallyVarying;
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("turbidity", m_turbidity, +ParamFlags::Differentiable);
        callback->put_parameter("sky_scale", m_sky_scale, +ParamFlags::NonDifferentiable);
        callback->put_parameter("sun_scale", m_sun_scale, +ParamFlags::NonDifferentiable);
        callback->put_parameter("albedo", m_albedo, +ParamFlags::NonDifferentiable);
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

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        Vector3f local_wi = dr::normalize(m_to_world.value().inverse().transform_affine(si.wi));
        Float cos_theta = Frame3f::cos_theta(-local_wi),
              cos_gamma = Frame3f::cos_theta(m_local_sun_frame.to_local(-local_wi));

        active &= cos_theta >= 0;

        UnpolarizedSpectrum res = 0.f;
        if constexpr (is_rgb_v<Spectrum>) {
            // dr::width(idx) == 1
            const SpecUInt32 idx = SpecUInt32({0, 1, 2});

            res = m_sky_scale * render_sky(idx, cos_theta, cos_gamma, active);
            res += m_sun_scale * render_sun(idx, cos_theta, cos_gamma, active) * 465.382521163; // FIXME: explain this factor

            res *= MI_CIE_Y_NORMALIZATION;

        } else {
            const Spectrum normalized_wavelengths = (si.wavelengths - WAVELENGTHS<ScalarFloat>[0]) / WAVELENGTH_STEP;

            const SpecUInt32 query_idx_low = dr::floor2int<SpecUInt32>(normalized_wavelengths),
                             query_idx_high = dr::minimum(query_idx_low + 1, NB_CHANNELS - 1);

            const Spectrum lerp_factor = normalized_wavelengths - query_idx_low;

            SpecMask spec_mask = active & (query_idx_low >= 0) & (query_idx_low < NB_CHANNELS);

            res = m_sky_scale * dr::lerp(
                render_sky(query_idx_low, cos_theta, cos_gamma, spec_mask),
                render_sky(query_idx_high, cos_theta, cos_gamma, spec_mask),
                lerp_factor); // FIXME: explain this factor * 465.382521163

            res += m_sun_scale * dr::lerp(
                render_sun(query_idx_low, cos_theta, cos_gamma, spec_mask),
                render_sun(query_idx_high, cos_theta, cos_gamma, spec_mask),
                lerp_factor);

            res *= m_d65->eval(si, active);
        }

        return dr::select(active & (res >= 0.f), res, 0.f);
    }


    std::pair<DirectionSample3f, Spectrum> sample_direction(const Interaction3f &it,
                                                            const Point2f &sample,
                                                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        // Sample the sun or the sky
        const ScalarFloat boundary = m_sky_scale / (m_sky_scale + m_sun_scale);
        auto [sample_dir, pdf] = dr::select(
                sample.x() < boundary,
                sample_sky({sample.x() / boundary, sample.y()}, active),
                sample_sun({(sample.x() - boundary) / (1 - boundary), sample.y()})
        );

        // Check for bounds on PDF
        Float sin_theta = Frame3f::sin_theta(sample_dir);
        active &= (Frame3f::cos_theta(sample_dir) >= 0.f) && (sin_theta != 0.f);
        sin_theta = dr::maximum(sin_theta, SIN_OFFSET);

        pdf = dr::select(active, pdf / sin_theta, 0.f);

        // Automatically enlarge the bounding sphere when it does not contain the reference point
        Float radius = dr::maximum(m_bsphere.radius, dr::norm(it.p - m_bsphere.center)),
              dist   = 2.f * radius;

        Vector3f d = m_to_world.value().transform_affine(sample_dir);
        DirectionSample3f ds;
        ds.p       = dr::fmadd(d, dist, it.p);
        ds.n       = -d;
        ds.uv      = sample;
        ds.time    = it.time;
        ds.pdf     = pdf;
        ds.delta   = false;
        ds.emitter = this;
        ds.d       = d;
        ds.dist    = dist;

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.wavelengths = it.wavelengths;

        return { ds, eval(si, active) / pdf };
    }

    Float pdf_direction(const Interaction3f &, const DirectionSample3f &ds,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        Vector3f local_dir = dr::normalize(m_to_world.value().inverse().transform_affine(ds.d));

        // Check for bounds on PDF
        Float sin_theta = Frame3f::sin_theta(local_dir);
        active &= (Frame3f::cos_theta(local_dir) >= 0.f) && (sin_theta != 0.f);
        sin_theta = dr::maximum(sin_theta, SIN_OFFSET);

        Float sun_pdf = warp::square_to_uniform_cone_pdf<true>(m_local_sun_frame.to_local(local_dir), SUN_COS_CUTOFF);
        Float sky_pdf = tgmm_pdf(from_spherical(local_dir), active) / sin_theta;

        Float combined_pdf = (m_sky_scale * sky_pdf + m_sun_scale * sun_pdf) / (m_sky_scale + m_sun_scale);

        return dr::select(active, combined_pdf, 0.f);
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
            << "  albedo = " << string::indent(m_albedo) << std::endl // TODO add `to_string`?
            << "]";
        return oss.str();
    }


    MI_DECLARE_CLASS()
private:

    Spectrum render_sky(const SpecUInt32& channel_idx,
        const Float& cos_theta, const Float& cos_gamma, const SpecMask& active) const {

        // Gather coefficients for the skylight equation
        Spectrum coefs[NB_SKY_PARAMS];
        for (uint8_t i = 0; i < NB_SKY_PARAMS; ++i)
            coefs[i] = dr::gather<Spectrum>(m_sky_params, channel_idx * NB_SKY_PARAMS + i, active);

        Float gamma = dr::safe_acos(cos_gamma),
              cos_gamma_sqr = dr::square(cos_gamma);

        Spectrum c1 = 1 + coefs[0] * dr::exp(coefs[1] / (cos_theta + 0.01f));
        Spectrum chi = (1 + cos_gamma_sqr) / dr::pow(1 + dr::square(coefs[8]) - 2 * coefs[8] * cos_gamma, 1.5);
        Spectrum c2 = coefs[2] + coefs[3] * dr::exp(coefs[4] * gamma) + coefs[5] * cos_gamma_sqr + coefs[6] * chi + coefs[7] * dr::safe_sqrt(cos_theta);

        return c1 * c2 * dr::gather<Spectrum>(m_sky_radiance, channel_idx, active);
    }

    Spectrum render_sun(const SpecUInt32& channel_idx, const Float& cos_theta, const Float& cos_gamma, const SpecMask& active) const {
        SpecMask hit_sun = active & (cos_gamma >= SUN_COS_CUTOFF);

        // Angles computation
        Float elevation =  0.5f * dr::Pi<Float> - dr::acos(cos_theta);
        Float sin_gamma_sqr      = dr::fnmadd(cos_theta, cos_theta, 1.f),
              sin_sun_cutoff_sqr = dr::fnmadd(SUN_COS_CUTOFF, SUN_COS_CUTOFF, 1.f);
        Float cos_phi = dr::safe_sqrt(1 - sin_gamma_sqr / sin_sun_cutoff_sqr);

        // Find the segment of the piecewise function we are in
        UInt32 pos = dr::floor2int<UInt32>(dr::pow(2 * elevation * dr::InvPi<Float>, 1.f/3.f) * NB_SUN_SEGMENTS);
        pos = dr::minimum(pos, NB_SUN_SEGMENTS - 1);

        const Float break_x = 0.5f * dr::Pi<Float> * dr::pow((Float)pos / NB_SUN_SEGMENTS, 3.f);
        const Float x = elevation - break_x;

        UnpolarizedSpectrum solar_radiance = 0.f;
        if constexpr (is_spectral_v<Spectrum>) {
            // Compute sun radiance
            UnpolarizedSpectrum sun_radiance = 0.f;
            SpecUInt32 global_idx = pos * NB_WAVELENGTHS * NB_SUN_CTRL_PTS + channel_idx * NB_SUN_CTRL_PTS;
            for (uint8_t k = 0; k < NB_SUN_CTRL_PTS; ++k)
                sun_radiance += dr::pow(x, k) * dr::gather<Spectrum>(m_sun_radiance, global_idx + k, hit_sun);


            // Compute limb darkening
            UnpolarizedSpectrum sun_ld = 0.f;
            global_idx = channel_idx * NB_SUN_LD_PARAMS;
            for (uint8_t j = 0; j < NB_SUN_LD_PARAMS; ++j)
                sun_ld += dr::pow(cos_phi, j) * dr::gather<Spectrum>(m_sun_ld, global_idx + j, hit_sun);

            solar_radiance = sun_ld * sun_radiance;

        } else {
            // Reproduces the spectral equation above but distributes the product of sums
            // since it uses interpolated coefficients from the spectral dataset

            SpecUInt32 global_idx = pos * (3 * NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS) +
                                    channel_idx * (NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS);

            // TODO use dr::pow ?
            Float x_exp = 1.f;
            for (uint8_t k = 0; k < NB_SUN_CTRL_PTS; ++k) {

                Float cos_exp = 1.f;
                for (uint8_t j = 0; j < NB_SUN_LD_PARAMS; ++j) {
                    solar_radiance += x_exp * cos_exp * dr::gather<Spectrum>(m_sun_radiance, global_idx + k * NB_SUN_LD_PARAMS + j, hit_sun);
                    cos_exp *= cos_phi;
                }

                x_exp *= x;
            }
        }

        // TODO negative values clamping (current not working)
        return dr::select(hit_sun & (solar_radiance > 0.f), solar_radiance, Spectrum(0.f));
    }

    MI_INLINE Point2f gaussian_cdf(const Point2f& mu, const Point2f& sigma, const Point2f& x) const {
        return 0.5f * (1 + dr::erf(dr::InvSqrtTwo<Float> * (x - mu) / sigma));
    }

    std::pair<Vector3f, Float> sample_sky(const Point2f& sample_, const Mask& active) const {
        // Sample a gaussian from the mixture
        const auto [idx, temp_sample] = m_gaussian_distr.sample_reuse(sample_.x(), active);

        // {mu_phi, mu_theta, sigma_phi, sigma_theta, weight}
        Gaussian gaussian = dr::gather<Gaussian>(m_gaussians, idx, active);

        const Point2f a = { 0.0 },
                      b = { dr::TwoPi<Float>, 0.5f * dr::Pi<Float> };

        const Point2f mu    = { gaussian[0], gaussian[1] },
                      sigma = { gaussian[2], gaussian[3] };

        const Point2f cdf_a = gaussian_cdf(mu, sigma, a),
                      cdf_b = gaussian_cdf(mu, sigma, b);

        Point2f sample = dr::lerp(cdf_a, cdf_b, Point2f{temp_sample, sample_.y()});
        // Clamp to erfinv's domain of definition
        sample = dr::clip(sample, dr::Epsilon<Float>, dr::OneMinusEpsilon<Float>);

        Point2f angles = dr::SqrtTwo<Float> * dr::erfinv(2 * sample - 1) * sigma + mu;

        // From fixed reference frame where sun_phi = pi/2 to local frame
        angles.x() += m_sun_phi - 0.5f * dr::Pi<Float>;
        // Clamp theta to avoid negative z-axis values (FP errors)
        angles.y() = dr::minimum(angles.y(), 0.5f * dr::Pi<Float> - dr::Epsilon<Float>);

        return { to_spherical(angles), tgmm_pdf(angles, active) };
    }

    std::pair<Vector3f, Float> sample_sun(const Point2f& sample) const {
        Vector3f sun_sample = warp::square_to_uniform_cone(sample, SUN_COS_CUTOFF);
        Float pdf = warp::square_to_uniform_cone_pdf(sun_sample, SUN_COS_CUTOFF);

        return { m_local_sun_frame.to_world(sun_sample), pdf };
    }

    Float tgmm_pdf(Point2f angles, Mask active) const {
        // From local frame to reference frame where sun_phi = pi/2 and phi is in [0, 2pi]
        angles.x() -= m_sun_phi - 0.5f * dr::Pi<Float>;
        angles.x() = dr::select(angles.x() < 0, angles.x() + dr::TwoPi<Float>, angles.x());

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

        return dr::select(active, pdf, 0.0);
    }

    FloatStorage extract_albedo(const ref<Texture>& albedo_tex) const {
        FloatStorage albedo = dr::zeros<FloatStorage>(NB_CHANNELS);
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

        if constexpr (is_rgb_v<Spectrum>) {
            // TODO find a way to load rgb buffer without loop
            Color3f temp = albedo_tex->eval(si);
            for (ScalarUInt32 i = 0; i < NB_CHANNELS; ++i)
                dr::scatter(albedo, temp[i], (UInt32) i);

        } else if constexpr (dr::is_array_v<Float> && is_spectral_v<Spectrum>) {
            si.wavelengths = dr::load<FloatStorage>(WAVELENGTHS<ScalarFloat>, NB_CHANNELS);
            albedo = albedo_tex->eval(si)[0];

        } else if (!dr::is_array_v<Float> && is_spectral_v<Spectrum>) {
            for (ScalarUInt32 i = 0; i < NB_CHANNELS; ++i) {
                si.wavelengths = WAVELENGTHS<ScalarFloat>[i];
                dr::scatter(albedo, albedo_tex->eval(si)[0], (UInt32) i);
            }

        } else {
            Throw("Unsupported spectrum type");
        }

        albedo = dr::clip(albedo, 0.f, 1.f);

        return albedo;
    }

    // ================================================================================================
    // ========================================= ATTRIBUTES ===========================================
    // ================================================================================================

    const std::string DATASET_NAME = is_spectral_v<Spectrum> ?
            DATABASE_PREFIX "_spec" :
            DATABASE_PREFIX "_rgb";

    /// Offset used to avoid division by zero in the pdf computation
    static constexpr ScalarFloat SIN_OFFSET = 0.00775;
    /// Number of channels used in the skylight model
    static constexpr uint32_t NB_CHANNELS = is_spectral_v<Spectrum> ? NB_WAVELENGTHS : 3;

    // Dataset sizes
    static constexpr uint32_t SKY_DATASET_SIZE = NB_TURBIDITY * NB_ALBEDO * NB_SKY_CTRL_PTS * NB_CHANNELS * NB_SKY_PARAMS,
                              SKY_DATASET_RAD_SIZE = NB_TURBIDITY * NB_ALBEDO * NB_SKY_CTRL_PTS * NB_CHANNELS,
                              SUN_DATASET_SIZE = NB_TURBIDITY * NB_CHANNELS * NB_SUN_SEGMENTS * NB_SUN_CTRL_PTS * (is_spectral_v<Spectrum> ? 1 : NB_SUN_LD_PARAMS),
                              TGMM_DATA_SIZE = (NB_TURBIDITY - 1) * NB_ETAS * NB_GAUSSIAN * NB_GAUSSIAN_PARAMS;

    /// Cosine of the sun's cutoff angle
    const Float SUN_COS_CUTOFF = (Float) dr::cos(0.5 * SUN_APERTURE * dr::Pi<double> / 180.0);

    Float m_surface_area;
    BoundingSphere3f m_bsphere;

    Float m_turbidity;
    ScalarFloat m_sky_scale;
    ScalarFloat m_sun_scale;
    ref<Texture> m_albedo;

    // Sun parameters
    Float m_sun_phi;
    Frame3f m_local_sun_frame;

    // Radiance parameters
    FloatStorage m_sky_params;
    FloatStorage m_sky_radiance;
    FloatStorage m_sun_radiance;

    // Sampling parameters
    FloatStorage m_gaussians;
    DiscreteDistribution<Float> m_gaussian_distr;

    ref<Texture> m_d65;

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

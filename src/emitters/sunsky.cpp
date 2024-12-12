#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>

#include <mitsuba/render/sunsky/sun_model.h>
#include <mitsuba/render/sunsky/sky_model.h>
#include <mitsuba/render/sunsky/sunsky_io.h>

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

#define NB_TURBIDITY 10
#define NB_ALBEDO 2
#define NB_CTRL_PTS 6
#define NB_PARAMS 9

#define NB_ETAS 30
#define NB_GAUSSIAN 5
#define NB_GAUSSIAN_PARAMS 5

#define DATABASE_PATH "sunsky-testing/res/datasets/"

template <typename Float, typename Spectrum>
class SunskyEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_to_world)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    using FloatStorage = DynamicBuffer<Float>;

    using Gaussian = dr::Array<Float, NB_GAUSSIAN_PARAMS>;
    using Albedo = std::array<ScalarFloat, is_spectral_v<Spectrum> ? 11 : 3>;
    using SolarRadiance = std::conditional_t<is_spectral_v<Spectrum>,
                                                ContinuousDistribution<Wavelength>,
                                                Color3f>;

    using SpecUInt32 = dr::uint32_array_t<Spectrum>;
    using SpecMask = dr::mask_t<Spectrum>;

    SunskyEmitter(const Properties &props) : Base(props) {
        /* Until `set_scene` is called, we have no information
           about the scene and default to the unit bounding sphere. */
        m_bsphere = BoundingSphere3f(ScalarPoint3f(0.f), 1.f);
        m_surface_area = 4.f * dr::Pi<Float>;

        m_d65 = Texture::D65(1.f);

        // ================ GET TURBIDITY ===============
        ScalarFloat turbidity = props.get<ScalarFloat>("turbidity", 3.f);

        // ================= GET ALBEDO =================
        ref<Texture> albedo = props.texture<Texture>("albedo", 1.f);
        if (albedo->is_spatially_varying())
            Throw("Expected a non-spatially varying radiance spectra!");

        dr::eval(albedo);
        Albedo albedo_buff = extract_albedo(albedo);

        // ================= GET ANGLES =================
        ScalarVector3f local_sun_dir = dr::normalize(
            m_to_world.scalar().inverse()
            .transform_affine(props.get<ScalarVector3f>("sun_direction")));

        m_local_sun_frame = Frame3f(local_sun_dir);

        ScalarFloat sun_phi = dr::atan2(local_sun_dir.y(), local_sun_dir.x());
        sun_phi = dr::select(sun_phi >= 0, sun_phi, sun_phi + dr::TwoPi<Float>);

        ScalarFloat sun_theta = dr::unit_angle_z(local_sun_dir),
                    sun_eta = 0.5f * dr::Pi<ScalarFloat> - sun_theta;

        // ================= GET SKY RADIANCE =================
        m_dataset = array_from_file<double, ScalarFloat>(DATABASE_PATH + DATASET_NAME + ".bin");
        m_rad_dataset = array_from_file<double, ScalarFloat>(DATABASE_PATH + DATASET_NAME + "_rad.bin");

        update_radiance_params(albedo_buff, turbidity, sun_eta);

        // ================= GET SUN RADIANCE =================
        update_sun_radiance(sun_theta, turbidity);

        // ================= GET TGMM TABLES =================
        m_tgmm_tables = array_from_file<float, ScalarFloat>(DATABASE_PATH "tgmm_tables.bin");

        update_tgmm_distribution(turbidity, sun_eta);


        m_turbidity = turbidity;
        m_sun_phi = sun_phi;
        m_local_sun_dir = local_sun_dir;

        dr::make_opaque(m_turbidity, m_sun_phi, m_local_sun_dir);

        m_flags = +EmitterFlags::Infinite | +EmitterFlags::SpatiallyVarying;
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("turbidity", m_turbidity, +ParamFlags::Differentiable);
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
        Float cos_theta = Frame3f::cos_theta(-local_wi);
        Float cos_gamma = dr::dot(m_local_sun_dir, -local_wi);

        active &= cos_theta >= 0;

        UnpolarizedSpectrum res = 0.f;
        if constexpr (is_rgb_v<Spectrum>) {
            // dr::width(idx) == 1
            SpecUInt32 idx = SpecUInt32({0, 1, 2});

            res = render_sky(idx, cos_theta, cos_gamma, active) * MI_CIE_Y_NORMALIZATION;

        } else {
            Spectrum normalized_wavelengths = (si.wavelengths - WAVELENGTHS[0]) / WAVELENGTH_STEP;
            SpecUInt32 query_idx = SpecUInt32(dr::floor(normalized_wavelengths));
            Spectrum lerp_factor = normalized_wavelengths - query_idx;

            SpecMask spec_mask = active & (query_idx >= 0) & (query_idx < NB_CHANNELS);

            res = dr::lerp(
                render_sky(query_idx, cos_theta, cos_gamma, spec_mask),
                render_sky(dr::minimum(query_idx + 1, NB_CHANNELS - 1), cos_theta, cos_gamma, spec_mask),
                lerp_factor) * m_d65->eval(si, active) * 465.382521163; // FIXME: explain this factor
        }

        res *= m_sky_scale;
        res += m_sun_scale * render_sun(cos_gamma, si.wavelengths, active);
        return dr::select(active & (res >= 0.f), res, 0.f);
    }

    std::pair<Vector3f, Float> sample_sun(const Point2f& sample) const {
        Vector3f sun_sample = warp::square_to_uniform_cone(sample, m_sun_cos_cutoff);
        sun_sample = m_local_sun_frame.to_world(sun_sample);

        return {
            sun_sample, warp::square_to_uniform_cone_pdf(sun_sample, m_sun_cos_cutoff)
        };
    }

    std::pair<Vector3f, Float> sample_sky(const Point2f& sample, const Mask& active) const {
        Vector3f sky_sample = tgmm_sample(sample, active);
        return {
            sky_sample, tgmm_pdf(sky_sample, active)
        };
    }


    std::pair<DirectionSample3f, Spectrum> sample_direction(const Interaction3f &it,
                                                            const Point2f &sample,
                                                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        ScalarFloat boundary = m_sky_scale / (m_sky_scale + m_sun_scale);
        auto [sample_dir, pdf] = dr::select(
                sample.x() < boundary,
                sample_sky({sample.x() / boundary, sample.y()}, active),
                sample_sun({(sample.x() - boundary) / (1 - boundary), sample.y()})
        );

        pdf = dr::select(sample_dir.z() < 0.f, 0.f, pdf * inv_sin_theta(sample_dir));

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

        Float sun_pdf = warp::square_to_uniform_cone_pdf(m_local_sun_frame.to_local(local_dir), m_sun_cos_cutoff);
        Float sky_pdf = tgmm_pdf(local_dir, active);

        return inv_sin_theta(local_dir) * (m_sky_scale * sky_pdf + m_sun_scale * sun_pdf) / (m_sky_scale + m_sun_scale);
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
            << "]";
        return oss.str();
    }


    MI_DECLARE_CLASS()
private:

    const std::string DATASET_NAME = is_spectral_v<Spectrum> ?
        "ssm_dataset_v2_spec" :
        "ssm_dataset_v2_rgb";

    static constexpr size_t WAVELENGTH_STEP = 40;
    static constexpr ScalarFloat WAVELENGTHS[11] = {
        320, 360, 400, 420, 460, 520, 560, 600, 640, 680, 720
    };

    static constexpr size_t NB_CHANNELS = is_spectral_v<Spectrum> ? 11 : 3,
                            DATASET_SIZE = NB_TURBIDITY * NB_ALBEDO * NB_CTRL_PTS * NB_CHANNELS * NB_PARAMS,
                            RAD_DATASET_SIZE = NB_TURBIDITY * NB_ALBEDO * NB_CTRL_PTS * NB_CHANNELS;

    Float m_surface_area;
    BoundingSphere3f m_bsphere;

    Float m_turbidity;
    ScalarFloat m_sky_scale = 1.f;
    ScalarFloat m_sun_scale = 0.f;

    Float m_sun_phi;
    Vector3f m_local_sun_dir;
    Frame3f m_local_sun_frame;
    Float m_sun_cos_cutoff = dr::cos(dr::deg_to_rad((ScalarFloat) (SUN_APP_RADIUS * 0.5)));

    ref<Texture> m_d65;

    // Radiance parameters
    FloatStorage m_params;
    FloatStorage m_sky_radiance;
    SolarRadiance m_sun_radiance;

    // Sampling parameters
    FloatStorage m_gaussians;
    DiscreteDistribution<Float> m_gaussian_distr;

    // Permanent datasets loaded from files/memory
    std::vector<ScalarFloat> m_dataset;
    std::vector<ScalarFloat> m_rad_dataset;
    std::vector<ScalarFloat> m_tgmm_tables;
    SunParameters<ScalarFloat> m_sun_params;


    Spectrum render_sky(const SpecUInt32& channel_idx,
        const Float& cos_theta, const Float& cos_gamma, const SpecMask& active) const {

        Spectrum coefs[NB_PARAMS];
        for (uint8_t i = 0; i < NB_PARAMS; ++i)
            coefs[i] = dr::gather<Spectrum>(m_params, channel_idx * NB_PARAMS + i, active);

        Float gamma = dr::acos(cos_gamma),
              cos_gamma_sqr = dr::square(cos_gamma);

        Spectrum c1 = 1 + coefs[0] * dr::exp(coefs[1] / (cos_theta + 0.01f));
        Spectrum chi = (1 + cos_gamma_sqr) / dr::pow(1 + dr::square(coefs[8]) - 2 * coefs[8] * cos_gamma, 1.5);
        Spectrum c2 = coefs[2] + coefs[3] * dr::exp(coefs[4] * gamma) + coefs[5] * cos_gamma_sqr + coefs[6] * chi + coefs[7] * dr::safe_sqrt(cos_theta);

        return c1 * c2 * dr::gather<Spectrum>(m_sky_radiance, channel_idx, active);
    }

    Spectrum render_sun(const Float& cos_gamma, const Wavelength& wavelengths, const Mask& active) const {
        Mask hit_sun = cos_gamma >= dr::cos(dr::deg_to_rad((ScalarFloat) (SUN_APP_RADIUS * 0.5)));

        Spectrum sun_radiance = 0.f;
        if constexpr (is_rgb_v<Spectrum>)
            sun_radiance = m_sun_radiance;
        else
            sun_radiance = m_sun_radiance.eval_pdf(wavelengths, active);

        return dr::select(active & hit_sun, sun_radiance, 0.f);
    }

    MI_INLINE Float inv_sin_theta(const Vector3f& local_dir) const {
        return dr::safe_rsqrt(dr::maximum(
            dr::square(local_dir.x()) + dr::square(local_dir.y()),
            dr::square(dr::Epsilon<Float>))
        );
    }

    MI_INLINE Point2f gaussian_cdf(const Point2f& mu, const Point2f& sigma, const Point2f& x) const {
        return 0.5f * (1 + dr::erf(dr::InvSqrtTwo<Float> * (x - mu) / sigma));
    }

    Float tgmm_pdf(const Vector3f& direction, Mask active) const {

        Point2f angles = from_spherical(direction);

        // From local frame to reference frame where sun_phi = pi/2
        angles.x() -= m_sun_phi - 0.5f * dr::Pi<Float>;
        angles.x() = dr::select(angles.x() < 0, angles.x() + dr::TwoPi<Float>, angles.x());

        // Bounds check for theta
        active &= (angles.y() > 0) && (angles.y() <= 0.5f * dr::Pi<Float>);

        const Point2f a = { 0.0 },
                      b = { dr::TwoPi<Float>, 0.5f * dr::Pi<Float> };

        Float pdf = 0.0;
        for (size_t i = 0; i < 4 * NB_GAUSSIAN; ++i) {
            const size_t base_idx = i * NB_GAUSSIAN_PARAMS;
            //Gaussian gaussian = dr::gather<Gaussian>(m_gaussians, base_idx, active);
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

    Vector3f tgmm_sample(const Point2f& sample_, const Mask& active) const {
        auto [idx, temp_sample] = m_gaussian_distr.sample_reuse(sample_.x(), active);

        Point2f sample = {temp_sample, sample_.y()};

        Gaussian gaussian = dr::gather<Gaussian>(m_gaussians, idx, active);

        const Point2f a = { 0.0 },
                      b = { dr::TwoPi<Float>, 0.5f * dr::Pi<Float> };

        const Point2f mu    = { gaussian[0], gaussian[1] },
                      sigma = { gaussian[2], gaussian[3] };

        const Point2f cdf_a = gaussian_cdf(mu, sigma, a),
                      cdf_b = gaussian_cdf(mu, sigma, b);

        sample = dr::lerp(cdf_a, cdf_b, sample);
        Point2f angles = dr::SqrtTwo<Float> * dr::erfinv(2 * sample - 1) * sigma + mu;
        // From fixed reference frame where sun_phi = pi/2 to local frame
        angles.x() += m_sun_phi - 0.5f * dr::Pi<Float>;

        return dr::normalize(to_spherical(angles));
    }

    void update_sun_radiance(ScalarFloat sun_theta, ScalarFloat turbidity) {
        std::vector<ScalarFloat> sun_radiance = compute_sun_radiance(m_sun_params, sun_theta, turbidity);

        //ScalarFloat half_aperture = dr::deg_to_rad(SUN_APP_RADIUS * 0.5),
        //            solid_angle = dr::TwoPi<Float> * (1 - dr::cos(half_aperture));

        // TODO: no need to multiply with solid angles since we work with radiance?
        // for (ScalarFloat& value : sun_radiance)
        //     value *= solid_angle;

        if constexpr (is_spectral_v<Spectrum>) {
            m_sun_radiance = SolarRadiance({350.f, 800.f},
                                    sun_radiance.data(),
                                    sun_radiance.size());

        } else if constexpr (is_rgb_v<Spectrum>) {
            FloatStorage value = dr::load<FloatStorage>(sun_radiance.data(), sun_radiance.size()),
                         wavelengths = dr::load<FloatStorage>(solarWavelenghts, 91);

            // Transform solar spectrum computed on 91 wavelengths to RGB
            Color<FloatStorage, 3> rgb = linear_rgb_rec(wavelengths);
            m_sun_radiance = { dr::mean(rgb.x() * value),
                               dr::mean(rgb.y() * value),
                               dr::mean(rgb.z() * value) };
            m_sun_radiance *= MI_CIE_Y_NORMALIZATION;

        } else {
            Throw("Unsupported spectrum type");
        }

        dr::make_opaque(m_sun_radiance);
    }

    void update_tgmm_distribution(ScalarFloat t, ScalarFloat eta) {

        eta = dr::rad_to_deg(eta);
        ScalarFloat eta_idx_f = dr::clip((eta - 2) / 3, 0, NB_ETAS - 1),
                    t_idx_f = dr::clip(t - 2, 0, (NB_TURBIDITY - 1) - 1);

        ScalarUInt32 eta_idx_low = dr::floor2int<ScalarUInt32>(eta_idx_f),
                     t_idx_low = dr::floor2int<ScalarUInt32>(t_idx_f);

        ScalarUInt32 eta_idx_high = dr::minimum(eta_idx_low + 1, NB_ETAS - 1),
                     t_idx_high = dr::minimum(t_idx_low + 1, (NB_TURBIDITY - 1) - 1);

        ScalarFloat eta_rem = eta_idx_f - eta_idx_low,
                    t_rem = t_idx_f - t_idx_low;

        const size_t t_block_size = m_tgmm_tables.size() / (NB_TURBIDITY - 1),
                     eta_block_size = t_block_size / NB_ETAS;

        const ScalarUInt64 indices[4] = {
            t_idx_low * t_block_size + eta_idx_low * eta_block_size,
            t_idx_low * t_block_size + eta_idx_high * eta_block_size,
            t_idx_high * t_block_size + eta_idx_low * eta_block_size,
            t_idx_high * t_block_size + eta_idx_high * eta_block_size
        };
        const ScalarFloat lerp_factors[4] = {
            (1 - t_rem) * (1 - eta_rem),
            (1 - t_rem) * eta_rem,
            t_rem * (1 - eta_rem),
            t_rem * eta_rem
        };
        std::vector<ScalarFloat> distrib_params(4 * eta_block_size);
        for (size_t mixture_idx = 0; mixture_idx < 4; ++mixture_idx) {
            for (size_t param_idx = 0; param_idx < eta_block_size; ++param_idx) {
                ScalarUInt32 index = mixture_idx * eta_block_size + param_idx;
                distrib_params[index] = m_tgmm_tables[indices[mixture_idx] + param_idx];
                distrib_params[index] *= index % NB_GAUSSIAN_PARAMS == 4 ? lerp_factors[mixture_idx] : 1;
            }
        }

        std::vector<ScalarFloat> mis_weights(4 * NB_GAUSSIAN);
        for (size_t gaussian_idx = 0; gaussian_idx < 4 * NB_GAUSSIAN; ++gaussian_idx)
            mis_weights[gaussian_idx] = distrib_params[gaussian_idx * NB_GAUSSIAN_PARAMS + 4];

        m_gaussians = dr::load<FloatStorage>(distrib_params.data(), distrib_params.size());
        m_gaussian_distr = DiscreteDistribution<Float>(mis_weights.data(), mis_weights.size());

        dr::make_opaque(m_gaussians, m_gaussian_distr);
    }

    void update_radiance_params(const Albedo& albedo,
                                ScalarFloat turbidity, ScalarFloat eta) {
        std::vector<ScalarFloat>
                params = compute_radiance_params(m_dataset, albedo, turbidity, eta),
                radiance = compute_radiance_params(m_rad_dataset, albedo, turbidity, eta);

        m_params = dr::load<FloatStorage>(params.data(), params.size());
        m_sky_radiance = dr::load<FloatStorage>(radiance.data(), radiance.size());

        dr::make_opaque(m_params, m_sky_radiance);
    }

    Albedo extract_albedo(const ref<Texture>& albedo) const {
        Albedo albedo_buff = {};
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        if constexpr (!dr::is_array_v<Float> && is_rgb_v<Spectrum>) {
            Color3f temp = albedo->eval(si);
            for (size_t i = 0; i < NB_CHANNELS; ++i)
                albedo_buff[i] = temp[i];

        } else if constexpr (!dr::is_array_v<Float> && is_spectral_v<Spectrum>) {
            for (size_t i = 0; i < NB_CHANNELS; ++i) {
                si.wavelengths = WAVELENGTHS[i];
                albedo_buff[i] = albedo->eval(si)[0];
            }

        } else if constexpr (is_rgb_v<Spectrum>) {
            Color3f temp = albedo->eval(si);
            for (size_t i = 0; i < NB_CHANNELS; ++i)
                albedo_buff[i] = temp[i][0];


        } else if constexpr (is_spectral_v<Spectrum>) {
            FloatStorage wavelengths = dr::load<FloatStorage>(WAVELENGTHS, NB_CHANNELS);
            si.wavelengths = wavelengths;
            Spectrum res = albedo->eval(si);

            dr::eval(res);
            Float&& temp = dr::migrate(res[0], AllocType::Host);
            dr::sync_thread();

            for (size_t i = 0; i < NB_CHANNELS; ++i)
                albedo_buff[i] = temp[i];

        } else {
            Log(Error, "Unsupported spectrum type");
        }

        for (size_t i = 0; i < NB_CHANNELS; ++i)
            albedo_buff[i] = dr::clip(albedo_buff[i], 0.f, 1.f);

        return albedo_buff;
    }

};

MI_IMPLEMENT_CLASS_VARIANT(SunskyEmitter, Emitter)
MI_EXPORT_PLUGIN(SunskyEmitter, "Sun and Sky dome background emitter")
NAMESPACE_END(mitsuba)
